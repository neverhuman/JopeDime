use super::*;

pub(crate) fn write_production_canary_intents(
    args: &Args,
    intent: &ActiveIntent,
    candidate: &Sha256Digest,
) -> Result<()> {
    schedule_promotion_run(
        args,
        intent.trainer.task,
        &intent.trainer.accepted_parent_checkpoint_sha256,
        candidate,
        intent.trainer.seed,
        0,
    )
}

pub(crate) fn schedule_promotion_run(
    args: &Args,
    task: Task,
    accepted_parent: &Sha256Digest,
    candidate: &Sha256Digest,
    base_seed: u64,
    run_index: usize,
) -> Result<()> {
    ensure!(run_index < 3, "promotion run index must be 0..3");
    let file_name = format!("{}-{}-run{}.v3.json", task.as_str(), candidate, run_index);
    let pending_path = args
        .root
        .join("training/task-promotion-evaluations/pending")
        .join(&file_name);
    let consumed_intent_path = args
        .root
        .join("training/task-promotion-evaluations/intents")
        .join(&file_name);
    if pending_path.is_file() || consumed_intent_path.is_file() {
        let path = if pending_path.is_file() {
            &pending_path
        } else {
            &consumed_intent_path
        };
        let existing: TaskPromotionEvaluationIntentV3 = serde_json::from_value(read_json(path)?)?;
        existing.validate()?;
        ensure!(
            existing.task == task
                && &existing.accepted_parent_checkpoint_sha256 == accepted_parent
                && &existing.candidate_checkpoint_sha256 == candidate
                && existing.run_index == run_index,
            "existing promotion intent scope drift"
        );
        return Ok(());
    }
    let pool: SealedPanelPoolV2 = serde_json::from_value(read_json(
        &args.root.join("training/sealed-panel-pool.v2.json"),
    )?)?;
    pool.validate()?;
    let consumed_directory = args.root.join("training/sealed-panels/consumed");
    let mut prior_with_paths = if consumed_directory.is_dir() {
        fs::read_dir(&consumed_directory)?
            .filter_map(|entry| entry.ok())
            .filter_map(|entry| {
                let path = entry.path();
                let receipt =
                    serde_json::from_value::<SealedPanelConsumptionV1>(read_json(&path).ok()?)
                        .ok()?;
                Some((path, receipt))
            })
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let prior = prior_with_paths
        .iter()
        .map(|(_, receipt)| receipt.clone())
        .collect::<Vec<_>>();
    let existing_consumption = prior_with_paths.iter().find(|(_, receipt)| {
        receipt.task == task
            && &receipt.candidate_checkpoint_sha256 == candidate
            && receipt.run_index == run_index
    });
    let (consumption_path, consumption) = if let Some((path, receipt)) = existing_consumption {
        validate_panel_consumption(&pool, receipt, &[])?;
        (path.clone(), receipt.clone())
    } else {
        let consumed_panels = prior
            .iter()
            .map(|receipt| receipt.panel.as_str())
            .collect::<BTreeSet<_>>();
        let panel = pool
            .entries
            .iter()
            .filter(|entry| entry.task == task)
            .filter_map(|entry| entry.promotion_panel.as_deref())
            .filter(|panel| !consumed_panels.contains(panel))
            .collect::<BTreeSet<_>>()
            .into_iter()
            .find(|panel| pool.promotion_panel(task, panel).len() >= 128)
            .context("sealed pool cannot supply the next unused task-local promotion panel")?;
        let available = pool.promotion_panel(task, panel);
        let scheduled = (available.len().min(512) - 128) / 64 * 64 + 128;
        let contexts = available
            .into_iter()
            .take(scheduled)
            .map(|context| context.context_sha256.clone())
            .collect::<Vec<_>>();
        let consumption = SealedPanelConsumptionV1 {
            schema: SEALED_PANEL_CONSUMPTION_V1_SCHEMA.to_owned(),
            task,
            panel: panel.to_owned(),
            candidate_checkpoint_sha256: candidate.clone(),
            run_index,
            seed: base_seed.saturating_add(run_index as u64 + 1),
            context_sha256: contexts.clone(),
            consumed_at_ms: now_ms(),
        };
        validate_panel_consumption(&pool, &consumption, &prior)?;
        let temporary = consumed_directory.join(format!(
            ".pending-{}-run{}.v1.json",
            candidate.as_str(),
            run_index
        ));
        atomic_json(&temporary, &serde_json::to_value(&consumption)?)?;
        let consumption_sha = sha256_file(&temporary)?;
        let consumption_path = consumed_directory.join(format!("{consumption_sha}.v1.json"));
        ensure!(
            !consumption_path.exists(),
            "sealed panel consumption identity already exists"
        );
        fs::rename(&temporary, &consumption_path)?;
        prior_with_paths.push((consumption_path.clone(), consumption.clone()));
        (consumption_path, consumption)
    };
    let consumption_sha = Sha256Digest::parse(sha256_file(&consumption_path)?)?;
    let promotion_intent = TaskPromotionEvaluationIntentV3 {
        schema: TASK_PROMOTION_EVALUATION_INTENT_V3_SCHEMA.to_owned(),
        task,
        accepted_parent_checkpoint_sha256: accepted_parent.clone(),
        candidate_checkpoint_sha256: candidate.clone(),
        run_index,
        seed: consumption.seed,
        panel_consumption: consumption_path,
        panel_consumption_sha256: consumption_sha,
        context_schedule: consumption
            .context_sha256
            .iter()
            .cloned()
            .map(|context_sha256| PromotionContextScheduleV3 { context_sha256 })
            .collect(),
        initial_contexts: 128,
        context_step: 64,
        maximum_contexts: consumption.context_sha256.len(),
        bootstrap_resamples: 10_000,
        created_at_ms: now_ms(),
    };
    promotion_intent.validate()?;
    atomic_json(&pending_path, &serde_json::to_value(promotion_intent)?)?;
    Ok(())
}

pub(crate) fn write_shadow_canary_intent(
    args: &Args,
    checkpoint: &ShadowCheckpointV1,
) -> Result<()> {
    let contexts = select_canary_contexts(
        &args.root,
        checkpoint.task,
        &checkpoint.accepted_parent_checkpoint_sha256,
        &checkpoint.checkpoint_sha256,
        0,
    )?;
    let rows = contexts
        .into_iter()
        .map(|context| {
            json!({
                "context_identity": context.context_identity,
                "class": context.class,
            })
        })
        .collect::<Vec<_>>();
    atomic_json(
        &args
            .root
            .join("training/shadow-canaries/pending")
            .join(format!(
                "{}-{}.v2.json",
                checkpoint.task.as_str(),
                checkpoint.checkpoint_sha256.as_str()
            )),
        &json!({
            "schema": "CanaryIntent.v2",
            "mode": "shadow",
            "task": checkpoint.task,
            "accepted_checkpoint_sha256":
                checkpoint.accepted_parent_checkpoint_sha256,
            "candidate_checkpoint_sha256": checkpoint.checkpoint_sha256,
            "run_index": 0,
            "seed": 20260726_u64,
            "contexts": rows,
            "candidate_proposals_per_context": 8,
            "accepted_baseline_proposals_per_context": 8,
            "zero_dl_gp_proposals_per_context": 8,
            "created_at_ms": now_ms(),
        }),
    )
}

pub(crate) fn select_canary_contexts(
    root: &Path,
    task: Task,
    accepted_parent: &Sha256Digest,
    candidate: &Sha256Digest,
    run_index: usize,
) -> Result<Vec<CanaryContextV2>> {
    let mut waves = fs::read_dir(root.join("waves"))?
        .filter_map(|entry| entry.ok())
        .filter_map(|entry| {
            let name = entry.file_name().to_string_lossy().into_owned();
            let number = name.strip_prefix("strict-wave-")?.parse::<u64>().ok()?;
            Some((number, entry.path()))
        })
        .collect::<Vec<_>>();
    waves.sort_by_key(|(number, _)| std::cmp::Reverse(*number));
    let mut frontier = BTreeSet::new();
    let mut hard = BTreeSet::new();
    let mut anchors = BTreeSet::new();
    for (_, path) in waves {
        let cohort_path = path.join("cohort.json");
        if !cohort_path.is_file() {
            continue;
        }
        let cohort = read_json(&cohort_path)?;
        if cohort.get("checkpoint_sha256").and_then(Value::as_str) != Some(accepted_parent.as_str())
        {
            continue;
        }
        let entries = cohort
            .get("entries")
            .and_then(Value::as_array)
            .context("cohort lacks entries")?;
        for row in entries {
            if row.get("task").and_then(Value::as_str) != Some(task.as_str()) {
                continue;
            }
            let Some(dataset) = row.get("dataset_hash").and_then(Value::as_str) else {
                continue;
            };
            match row.get("micro_lane").and_then(Value::as_str) {
                Some("hard_uncertain") => {
                    hard.insert(dataset.to_owned());
                }
                Some("stable_anchor") => {
                    anchors.insert(dataset.to_owned());
                }
                _ => {
                    frontier.insert(dataset.to_owned());
                }
            }
        }
        if frontier.len() >= 12 * (run_index + 1)
            && hard.len() >= 2 * (run_index + 1)
            && anchors.len() >= 2 * (run_index + 1)
        {
            return deterministic_canary_contexts(
                candidate,
                run_index,
                &frontier.into_iter().collect::<Vec<_>>(),
                &hard.into_iter().collect::<Vec<_>>(),
                &anchors.into_iter().collect::<Vec<_>>(),
            );
        }
    }
    bail!("no accepted-parent cohort can supply a 12/2/2 task canary")
}

pub(crate) fn find_train_result(
    runs_root: &Path,
    job_id: &str,
    output_name: &str,
) -> Result<Option<PathBuf>> {
    let attempts = runs_root.join(job_id).join("attempts");
    if !attempts.is_dir() {
        return Ok(None);
    }
    let mut paths = fs::read_dir(attempts)?
        .filter_map(|entry| entry.ok())
        .map(|entry| {
            entry
                .path()
                .join("outputs/training")
                .join(output_name)
                .join("train-result.json")
        })
        .filter(|path| path.is_file())
        .collect::<Vec<_>>();
    paths.sort();
    Ok(paths.pop())
}

pub(crate) fn admission_request_pending(root: &Path, job_id: &str) -> Result<bool> {
    let path = root.join("ops/gpu-lane-admission/requests/alternating_trainer.json");
    if !path.is_file() {
        return Ok(false);
    }
    Ok(read_json(&path)?
        .get("immutable_job_identity")
        .and_then(Value::as_str)
        == Some(job_id))
}

pub(crate) fn scheduler_jobs(value: &Value) -> Result<Vec<SchedulerJob>> {
    Ok(serde_json::from_value(
        value
            .get("Jobs")
            .and_then(|value| value.get("jobs"))
            .cloned()
            .context("SCQ response lacks Jobs.jobs")?,
    )?)
}

pub(crate) fn command_message(output: &std::process::Output) -> String {
    String::from_utf8_lossy(if output.stderr.is_empty() {
        &output.stdout
    } else {
        &output.stderr
    })
    .trim()
    .to_owned()
}

pub(crate) fn capacity_backoff(message: &str) -> bool {
    message.contains("dependency-unsatisfied")
        && (message.contains("no eligible launch-ready node currently")
            || message.contains("eligible_node launch-ready-capacity")
            || (message.contains("trainer GPU capability") && message.contains("exit 1")))
}

pub(crate) fn write_immutable(path: &Path, bytes: &[u8]) -> Result<()> {
    if path.is_file() {
        ensure!(
            sha256_file(path)? == sha256_bytes(bytes),
            "immutable trainer spec content drift"
        );
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let name = path
        .file_name()
        .and_then(|value| value.to_str())
        .context("immutable file has no name")?;
    let temporary = path.with_file_name(format!(".{name}.tmp-{}", std::process::id()));
    let mut file = OpenOptions::new()
        .create_new(true)
        .write(true)
        .mode(0o600)
        .open(&temporary)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    fs::rename(temporary, path)?;
    Ok(())
}

pub(crate) fn copy_immutable(source: &Path, target: &Path, digest: &str) -> Result<()> {
    if target.is_file() {
        ensure!(
            sha256_file(target)? == digest,
            "immutable checkpoint target content drift"
        );
        return Ok(());
    }
    let temporary = target.with_extension(format!("tmp-{}", std::process::id()));
    fs::copy(source, &temporary)?;
    ensure!(
        sha256_file(&temporary)? == digest,
        "copied checkpoint hash drift"
    );
    fs::rename(temporary, target)?;
    Ok(())
}

pub(crate) fn append_event<T: Serialize>(root: &Path, kind: &str, detail: &T) -> Result<()> {
    let path = root.join("ops/task-learning-events.v2.jsonl");
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut file = OpenOptions::new().create(true).append(true).open(path)?;
    serde_json::to_writer(
        &mut file,
        &json!({
            "schema": "TaskLearningControllerEvent.v2",
            "kind": kind,
            "created_at_ms": now_ms(),
            "detail": detail,
        }),
    )?;
    file.write_all(b"\n")?;
    file.sync_data()?;
    Ok(())
}

pub(crate) fn status(
    args: &Args,
    state: &ControllerState,
    accepted: &BTreeMap<Task, AcceptedPointer>,
    corpus: &CorpusPointer,
    action: &str,
    detail: Option<&str>,
) -> Value {
    json!({
        "schema": "TaskLearningControllerStatus.v2",
        "generated_at_ms": now_ms(),
        "mode": if args.submit { "submit" } else { "audit" },
        "action": action,
        "detail": detail,
        "accepted_by_task": accepted.iter().map(|(task, pointer)| {
            (task.as_str(), pointer.digest.as_str())
        }).collect::<BTreeMap<_, _>>(),
        "corpus_watermark": corpus.watermark,
        "intent": state.intent,
        "shadow_models": state.shadow_models,
        "last_shadow_corpus_watermark": state.last_shadow_corpus_watermark,
        "last_recipe_e_corpus_watermark": state.last_recipe_e_corpus_watermark,
        "last_recipe_e_descendant_receipts":
            state.last_recipe_e_descendant_receipts,
        "production_candidates": state.production_candidates,
        "legacy_monolithic_promotion_enabled": false,
    })
}
