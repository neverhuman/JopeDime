use super::*;

pub(super) fn explain_checkpoint(args: &Args) -> Result<()> {
    let lineage = scan_lineage(&args.runs_root);
    let candidates = checkpoint_candidates(&args.root, &lineage);
    let selected = select_checkpoint(&candidates);
    let report = json!({
        "schema": "JopeContinuousTrainerWatchdogCheckpointPlan.v1",
        "generated_at_ms": now_ms() as u64,
        "runs_root": args.runs_root.to_string_lossy(),
        "checkpoint_schemas": CHECKPOINT_SCHEMAS
            .iter()
            .map(|schema| json!({
                "name": schema.name,
                "marker": schema.marker,
                "required_tensors": schema.tensors,
                "required_groups": schema.groups,
            }))
            .collect::<Vec<_>>(),
        "lineage_runs": lineage
            .iter()
            .map(|run| json!({"job_id": run.job_id, "checkpoint": run.checkpoint.describe()}))
            .collect::<Vec<_>>(),
        "candidates": candidates.iter().map(Checkpoint::describe).collect::<Vec<_>>(),
        "selected": selected.as_ref().ok().map(Checkpoint::describe),
        "error": selected.as_ref().err().map(|error| format!("{error:#}")),
    });
    println!("{}", serde_json::to_string_pretty(&report)?);
    selected.map(|_| ())
}

pub(super) fn cycle(args: &Args, lineage: &mut LineageCache) -> Result<CycleReport> {
    let state_path = args.root.join("ops/trainer-watchdog-state.v1.json");
    let mut state = load_state(&state_path)?;
    let labels = load_label_watermark(&args.label_pointer())?;
    ensure!(
        labels >= state.max_observed_labels,
        "label pointer regressed from {} to {}",
        state.max_observed_labels,
        labels
    );
    state.max_observed_labels = labels;

    let jobs = parse_jobs(&scq_json(&args.scq, &["ls"])?)?;
    let priority = parse_priority(&scq_json(&args.scq, &["priority", "ls"])?)?;
    let trainers = jobs
        .iter()
        .filter_map(parse_trainer_job)
        .collect::<Result<Vec<_>>>()?;
    let active = active_trainers(&trainers)?;

    let completed_watermark = trainers
        .iter()
        .filter(|job| job.state == "completed")
        .map(|job| job.watermark)
        .max()
        .unwrap_or(0);
    state.last_evidence_watermark = state.last_evidence_watermark.max(completed_watermark);
    let observed_sequence = trainers.iter().map(|job| job.sequence).max().unwrap_or(0);
    state.next_sequence = state.next_sequence.max(observed_sequence.saturating_add(1));

    if let Some(intent) = &state.intent {
        if let Some(job) = trainers.iter().find(|job| job.id == intent.job_id) {
            if job.state == "completed" {
                state.last_evidence_watermark = state.last_evidence_watermark.max(intent.watermark);
            }
            state.intent = None;
        }
    }

    let next_label_threshold = state.last_evidence_watermark.saturating_add(LABEL_STEP);

    if let Some(job) = active.first() {
        if let Some(intent) = state.intent.take() {
            append_event(
                &args.root,
                "intent_superseded_by_active_trainer",
                json!({
                    "job_id": intent.job_id,
                    "active_job_id": job.id,
                    "spec_sha256": intent.spec_sha256,
                }),
            )?;
        }
        // Do not repeatedly hash the large checkpoint and packed corpus while
        // an active trainer already blocks admission. Full content validation
        // is mandatory immediately before rendering or retrying a submission.
        let checkpoint = load_checkpoint_reference(&args.root, &args.runs_root, lineage)?;
        let lineage_full_runs = lineage
            .get(&args.runs_root, false)
            .iter()
            .filter(|run| run.checkpoint.is_full())
            .count();
        let snapshot_watermark = load_snapshot_watermark(&args.corpus_pointer(), labels)?;
        ensure!(
            snapshot_watermark >= state.max_observed_snapshot_watermark,
            "training corpus pointer regressed from {} to {}",
            state.max_observed_snapshot_watermark,
            snapshot_watermark
        );
        state.max_observed_snapshot_watermark = snapshot_watermark;
        state.updated_at_ms = now_ms();
        save_state(&state_path, &state)?;
        return Ok(CycleReport {
            schema: WATCHDOG_SCHEMA,
            generated_at_ms: now_ms(),
            mode: if args.submit { "submit" } else { "audit" },
            continuous: true,
            label_step: LABEL_STEP,
            labels,
            snapshot_watermark,
            pending_unfrozen_labels: labels.saturating_sub(snapshot_watermark),
            last_evidence_watermark: state.last_evidence_watermark,
            next_label_threshold,
            checkpoint_sha256: checkpoint.sha256.clone(),
            checkpoint_origin: checkpoint.origin.clone(),
            checkpoint_tensors: checkpoint.tensors,
            checkpoint_full: checkpoint.is_full(),
            lineage_full_runs,
            family_priority: priority,
            priority_ready: priority_is_ready(priority),
            active_trainers: vec![job.id.clone()],
            action: "active_trainer_preserved".to_owned(),
            planned_job_id: None,
            continuation_kind: None,
            submitted: false,
            protected_work_mutations: 0,
            promotion_mutations: 0,
        });
    }

    ensure!(
        priority_is_ready(priority),
        "{PROJECT} effective priority is {priority}, expected {REQUIRED_PRIORITY}"
    );
    let checkpoint = load_checkpoint(&args.root, &args.runs_root, lineage)?;
    let lineage_full_runs = lineage
        .get(&args.runs_root, false)
        .iter()
        .filter(|run| run.checkpoint.is_full())
        .count();
    let snapshot = load_training_snapshot(&args.corpus_pointer(), labels)?;
    ensure!(
        snapshot.watermark >= state.max_observed_snapshot_watermark,
        "training corpus pointer regressed from {} to {}",
        state.max_observed_snapshot_watermark,
        snapshot.watermark
    );
    state.max_observed_snapshot_watermark = snapshot.watermark;
    let pending_unfrozen_labels = labels.saturating_sub(snapshot.watermark);
    let kind = continuation_kind(snapshot.watermark, state.last_evidence_watermark);

    // Specs from an older contract either stage the 9+ GiB source JSONL files
    // or reserve CPU capacity that SCQ cannot currently admit despite an idle
    // GPU and bounded physical CPU headroom. Retire such an unsubmitted intent
    // explicitly and create a new immutable job identity below.
    if state
        .intent
        .as_ref()
        .is_some_and(intent_uses_stale_contract)
    {
        let stale = state
            .intent
            .take()
            .context("legacy trainer intent disappeared")?;
        append_event(
            &args.root,
            "intent_superseded_stale_spec_contract",
            json!({
                "job_id": stale.job_id,
                "spec_sha256": stale.spec_sha256,
                "checkpoint_sha256": stale.checkpoint_sha256,
            }),
        )?;
        state.updated_at_ms = now_ms();
        save_state(&state_path, &state)?;
    }

    let intent_checkpoint_disposition = state
        .intent
        .as_mut()
        .map(|intent| observe_intent_checkpoint(intent, &checkpoint.sha256))
        .unwrap_or(IntentCheckpointDisposition::Current);
    if let IntentCheckpointDisposition::Confirming(confirmation) = intent_checkpoint_disposition {
        state.updated_at_ms = now_ms();
        save_state(&state_path, &state)?;
        bail!(
            "pending trainer intent checkpoint is stale; confirmation {confirmation}/{STALE_INTENT_CONFIRMATIONS} before superseding confirmed-unsubmitted intent"
        );
    }
    if intent_checkpoint_disposition == IntentCheckpointDisposition::Supersede {
        let stale = state
            .intent
            .take()
            .context("stale trainer intent disappeared")?;
        append_event(
            &args.root,
            "intent_superseded_unsubmitted",
            json!({
                "job_id": stale.job_id,
                "old_checkpoint_sha256": stale.checkpoint_sha256,
                "current_checkpoint_sha256": checkpoint.sha256,
                "spec_sha256": stale.spec_sha256,
                "confirmation_samples": STALE_INTENT_CONFIRMATIONS,
            }),
        )?;
        state.updated_at_ms = now_ms();
        save_state(&state_path, &state)?;
    }

    let intent = match state.intent.clone() {
        Some(intent) => {
            validate_intent(&intent, &snapshot, &checkpoint)?;
            intent
        }
        None => {
            let sequence = state.next_sequence;
            let seed = args
                .seed_base
                .checked_add(sequence)
                .context("trainer seed overflow")?;
            let job_id = trainer_job_id(snapshot.watermark, &args.recipe, sequence);
            let spec = args.root.join("specs").join(format!("{job_id}.toml"));
            let content = render_spec(args, &job_id, seed, &snapshot, &checkpoint)?;
            let intent = SubmissionIntent {
                job_id,
                watermark: snapshot.watermark,
                sequence,
                checkpoint_sha256: checkpoint.sha256.clone(),
                spec,
                spec_sha256: sha256_bytes(content.as_bytes()),
                mode: kind.as_str().to_owned(),
                seed,
                created_at_ms: now_ms(),
                last_error: None,
                checkpoint_mismatch_observations: 0,
            };
            if args.submit {
                write_immutable(&intent.spec, content.as_bytes())?;
                state.intent = Some(intent.clone());
                state.next_sequence = sequence.saturating_add(1);
                state.updated_at_ms = now_ms();
                save_state(&state_path, &state)?;
                append_event(
                    &args.root,
                    "submission_intent",
                    json!({
                        "job_id": intent.job_id,
                        "watermark": intent.watermark,
                        "sequence": intent.sequence,
                        "checkpoint_sha256": intent.checkpoint_sha256,
                        "checkpoint": checkpoint.describe(),
                        "spec_sha256": intent.spec_sha256,
                        "mode": intent.mode,
                    }),
                )?;
            }
            intent
        }
    };

    let submitted = if args.submit {
        let content_hash = sha256_file(&intent.spec)?;
        ensure!(
            content_hash == intent.spec_sha256,
            "durable trainer spec content drift"
        );
        match scq_check_submit(&args.scq, &intent.spec) {
            Ok(()) => {
                append_event(
                    &args.root,
                    "submission_accepted",
                    json!({"job_id": intent.job_id, "spec_sha256": intent.spec_sha256}),
                )?;
                true
            }
            Err(error) => {
                if let Some(saved) = state.intent.as_mut() {
                    saved.last_error = Some(format!("{error:#}"));
                }
                state.updated_at_ms = now_ms();
                save_state(&state_path, &state)?;
                return Err(error);
            }
        }
    } else {
        false
    };
    state.updated_at_ms = now_ms();
    save_state(&state_path, &state)?;

    Ok(CycleReport {
        schema: WATCHDOG_SCHEMA,
        generated_at_ms: now_ms(),
        mode: if args.submit { "submit" } else { "audit" },
        continuous: true,
        label_step: LABEL_STEP,
        labels,
        snapshot_watermark: snapshot.watermark,
        pending_unfrozen_labels,
        last_evidence_watermark: state.last_evidence_watermark,
        next_label_threshold,
        checkpoint_sha256: checkpoint.sha256.clone(),
        checkpoint_origin: checkpoint.origin.clone(),
        checkpoint_tensors: checkpoint.tensors,
        checkpoint_full: checkpoint.is_full(),
        lineage_full_runs,
        family_priority: priority,
        priority_ready: true,
        active_trainers: Vec::new(),
        action: if submitted {
            "submitted_continuous_trainer"
        } else {
            "would_submit_continuous_trainer"
        }
        .to_owned(),
        planned_job_id: Some(intent.job_id.clone()),
        continuation_kind: Some(intent.mode),
        submitted,
        protected_work_mutations: 0,
        promotion_mutations: 0,
    })
}
