use super::*;

pub(super) fn load_state(path: &Path) -> Result<ControllerState> {
    if !path.is_file() {
        return Ok(ControllerState::default());
    }
    let state: ControllerState = serde_json::from_value(read_json(path)?)?;
    ensure!(
        state.schema == STATE_SCHEMA,
        "task learning state schema drift"
    );
    if let Some(intent) = &state.intent {
        intent.trainer.validate()?;
        ensure!(
            intent.spec_path.is_file()
                && sha256_file(&intent.spec_path)? == intent.trainer.immutable_spec_sha256.as_str(),
            "persisted task trainer intent spec drift"
        );
    }
    Ok(state)
}

pub(super) fn save_state(path: &Path, state: &ControllerState) -> Result<()> {
    ensure!(
        state.schema == STATE_SCHEMA,
        "task learning state schema drift"
    );
    atomic_json(path, &serde_json::to_value(state)?)
}

pub(super) fn accepted_pointers(root: &Path) -> Result<BTreeMap<Task, AcceptedPointer>> {
    let value = read_json(&root.join("training/accepted-by-task.json"))?;
    ensure!(
        value.get("schema").and_then(Value::as_str) == Some("JopeSelfplayAcceptedByTask.v1"),
        "accepted-by-task pointer schema drift"
    );
    let mut output = BTreeMap::new();
    for task in Task::ALL {
        let row = match value
            .get("tasks")
            .and_then(|tasks| tasks.get(task.as_str()))
        {
            Some(row) => row,
            // Legacy tasks must always be present; the new tasks appear in
            // accepted-by-task.json only once their first anchor is promoted.
            None if matches!(task, Task::Binary | Task::Regression) => {
                bail!("accepted-by-task lacks {task}")
            }
            None => continue,
        };
        let path = PathBuf::from(
            row.get("checkpoint")
                .and_then(Value::as_str)
                .with_context(|| format!("{task} accepted checkpoint path is absent"))?,
        );
        let digest = Sha256Digest::parse(
            row.get("checkpoint_sha256")
                .and_then(Value::as_str)
                .with_context(|| format!("{task} accepted checkpoint digest is absent"))?
                .to_owned(),
        )?;
        ensure!(
            path.is_file() && sha256_file(&path)? == digest.as_str(),
            "{task} accepted checkpoint content drift"
        );
        output.insert(task, AcceptedPointer { digest, path });
    }
    Ok(output)
}

pub(super) fn bootstrap_evaluation_anchor(
    root: &Path,
    task: Task,
) -> Result<Option<AcceptedPointer>> {
    if matches!(task, Task::Binary | Task::Regression) {
        return Ok(None);
    }
    let anchor_path = root
        .join("training/bootstrap-evaluation-anchors")
        .join(format!("{}.v1.json", task.as_str()));
    if !anchor_path.is_file() {
        return Ok(None);
    }
    let anchor = read_json(&anchor_path)?;
    ensure!(
        anchor.get("schema").and_then(Value::as_str) == Some("TaskBootstrapEvaluationAnchor.v1")
            && anchor.get("task").and_then(Value::as_str) == Some(task.as_str())
            && anchor.get("accepted").and_then(Value::as_bool) == Some(false),
        "optional-task bootstrap evaluation anchor is not an unaccepted v1 anchor"
    );
    let path = PathBuf::from(
        anchor
            .get("checkpoint")
            .and_then(Value::as_str)
            .context("bootstrap evaluation anchor lacks checkpoint path")?,
    );
    let digest = Sha256Digest::parse(
        anchor
            .get("checkpoint_sha256")
            .and_then(Value::as_str)
            .context("bootstrap evaluation anchor lacks checkpoint digest")?
            .to_owned(),
    )?;
    ensure!(
        path.is_file() && sha256_file(&path)? == digest.as_str(),
        "bootstrap evaluation anchor content drift"
    );
    Ok(Some(AcceptedPointer { digest, path }))
}

pub(super) fn reset_changed_shadow_epochs(
    state: &mut ControllerState,
    accepted: &BTreeMap<Task, AcceptedPointer>,
) {
    for task in Task::ALL {
        if state.shadow_models.get(&task).is_some_and(|model| {
            accepted
                .get(&task)
                .is_none_or(|pointer| pointer.digest != model.accepted_parent_checkpoint_sha256)
        }) {
            state.shadow_models.remove(&task);
            state.last_shadow_corpus_watermark.remove(&task);
            state.last_recipe_e_corpus_watermark.remove(&task);
            state.last_recipe_e_descendant_receipts.remove(&task);
        }
    }
}

pub(super) fn apply_task_canary_acceptances(
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

pub(super) fn independent_external_passes(
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

pub(super) fn promote_task_pointer(
    args: &Args,
    task: Task,
    parent: &AcceptedPointer,
    child: &ModelRef,
    external_pass_receipt_ids: &[String],
) -> Result<()> {
    ensure!(
        external_pass_receipt_ids.len() == 3
            && external_pass_receipt_ids
                .iter()
                .collect::<BTreeSet<_>>()
                .len()
                == 3,
        "task promotion requires three independent external pass receipts"
    );
    ensure!(
        child.checkpoint_path.is_file()
            && sha256_file(&child.checkpoint_path)? == child.checkpoint_sha256.as_str(),
        "production child checkpoint content drift"
    );
    let pointer_path = args.root.join("training/accepted-by-task.json");
    let mut pointer = read_json(&pointer_path)?;
    let current = pointer
        .get("tasks")
        .and_then(|tasks| tasks.get(task.as_str()))
        .and_then(|row| row.get("checkpoint_sha256"))
        .and_then(Value::as_str)
        .map(str::to_owned);
    let absent_to_candidate = current.is_none();
    if absent_to_candidate {
        ensure!(
            matches!(task, Task::Multiclass | Task::Timeseries),
            "legacy task pointer cannot be absent during compare-and-swap"
        );
        let anchor = bootstrap_evaluation_anchor(&args.root, task)?
            .context("first optional-task promotion lacks a bootstrap evaluation anchor")?;
        ensure!(
            anchor.digest == parent.digest,
            "bootstrap evaluation anchor changed before absent-to-candidate compare-and-swap"
        );
    } else {
        ensure!(
            current.as_deref() == Some(parent.digest.as_str()),
            "accepted task pointer changed before compare-and-swap"
        );
    }
    let tasks = pointer
        .get_mut("tasks")
        .and_then(Value::as_object_mut)
        .context("accepted task pointer task map is not mutable")?;
    let row = tasks
        .entry(task.as_str().to_owned())
        .or_insert_with(|| json!({}))
        .as_object_mut()
        .context("accepted task pointer row is not mutable")?;
    row.insert(
        "checkpoint".to_owned(),
        json!(child.checkpoint_path.to_string_lossy()),
    );
    row.insert(
        "checkpoint_sha256".to_owned(),
        json!(child.checkpoint_sha256.as_str()),
    );
    row.insert("source".to_owned(), json!("task_aware_learning_controller"));
    row.insert(
        "promotion_tier".to_owned(),
        json!("three_independent_single_use_external_panels"),
    );
    row.insert("updated_at_ms".to_owned(), json!(now_ms()));
    let binary = pointer
        .get("tasks")
        .and_then(|tasks| tasks.get("binary"))
        .and_then(|row| row.get("checkpoint_sha256"))
        .and_then(Value::as_str);
    let regression = pointer
        .get("tasks")
        .and_then(|tasks| tasks.get("regression"))
        .and_then(|row| row.get("checkpoint_sha256"))
        .and_then(Value::as_str);
    pointer["global_checkpoint_sha256"] = if binary == regression {
        binary.map_or(Value::Null, |value| json!(value))
    } else {
        Value::Null
    };
    pointer["global_pointer"] = Value::Null;
    pointer["updated_at_ms"] = json!(now_ms());
    atomic_json(&pointer_path, &pointer)?;
    let receipt = json!({
        "schema": "TaskCheckpointPromotion.v2",
        "task": task,
        "parent_checkpoint_sha256": parent.digest,
        "child_checkpoint_sha256": child.checkpoint_sha256,
        "child_checkpoint": child.checkpoint_path,
        "external_pass_receipt_ids": external_pass_receipt_ids,
        "independent_single_use_panel_acceptances": 3,
        "compare_and_swap": if absent_to_candidate { "absent_to_candidate" } else { "parent_to_candidate" },
        "bootstrap_anchor_was_accepted": false,
        "evidence_counter_reset": true,
        "promoted_at_ms": now_ms(),
    });
    atomic_json(
        &args.root.join("training/task-promotions").join(format!(
            "{}-{}.v1.json",
            task.as_str(),
            child.checkpoint_sha256.as_str()
        )),
        &receipt,
    )?;
    atomic_json(
        &args.root.join("training/evidence-resets").join(format!(
            "{}-{}.v1.json",
            task.as_str(),
            child.checkpoint_sha256.as_str()
        )),
        &json!({
            "schema": "TaskEvidenceReset.v1",
            "task": task,
            "new_accepted_parent_checkpoint_sha256": child.checkpoint_sha256,
            "status": "awaiting_ten_preflight_cohorts",
            "proposer_namespace": format!(
                "{}-{}",
                task.as_str(),
                child.checkpoint_sha256.as_str()
            ),
            "reset_at_ms": now_ms(),
        }),
    )?;
    append_event(&args.root, "task_pointer_promoted", &receipt)
}

pub(super) fn corpus_pointer(root: &Path) -> Result<CorpusPointer> {
    let value = read_json(&root.join("training/current-corpus.json"))?;
    let manifest = PathBuf::from(
        value
            .get("manifest")
            .and_then(Value::as_str)
            .context("corpus pointer lacks manifest")?,
    );
    let packed_manifest = PathBuf::from(
        value
            .get("packed_manifest")
            .and_then(Value::as_str)
            .context("corpus pointer lacks packed manifest")?,
    );
    let pointer = CorpusPointer {
        watermark: value
            .get("watermark")
            .and_then(Value::as_u64)
            .context("corpus pointer lacks watermark")?,
        manifest,
        manifest_sha256: value
            .get("manifest_sha256")
            .and_then(Value::as_str)
            .context("corpus pointer lacks manifest digest")?
            .to_owned(),
        packed_manifest,
        packed_manifest_sha256: value
            .get("packed_manifest_sha256")
            .and_then(Value::as_str)
            .context("corpus pointer lacks packed manifest digest")?
            .to_owned(),
    };
    ensure!(
        pointer.manifest.is_file()
            && sha256_file(&pointer.manifest)? == pointer.manifest_sha256
            && pointer.packed_manifest.is_file()
            && sha256_file(&pointer.packed_manifest)? == pointer.packed_manifest_sha256,
        "task trainer corpus pointer content drift"
    );
    Ok(pointer)
}

pub(super) fn load_ledger(root: &Path) -> Result<Option<CausalEvidenceLedgerV3>> {
    let path = root.join("training/causal-evidence-ledger.v3.json");
    if !path.is_file() {
        return Ok(None);
    }
    let ledger: CausalEvidenceLedgerV3 = serde_json::from_value(read_json(&path)?)?;
    ledger.validate()?;
    Ok(Some(ledger))
}
