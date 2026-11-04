use super::*;

pub(crate) fn is_task_trainer_job_id(job_id: &str) -> bool {
    [
        "jope-selfplay-shadow-",
        "jope-selfplay-production-",
        "jope-selfplay-trainer-",
    ]
    .iter()
    .any(|prefix| job_id.starts_with(prefix))
}

pub(crate) fn trainer_spec_inputs(spec: &Path) -> Result<Vec<DimeNamedDigestV1>> {
    let body = fs::read_to_string(spec)?;
    let mut inputs = Vec::new();
    let mut name: Option<String> = None;
    let mut sha256: Option<String> = None;
    let flush = |name: &mut Option<String>,
                 sha256: &mut Option<String>,
                 inputs: &mut Vec<DimeNamedDigestV1>|
     -> Result<()> {
        if name.is_none() && sha256.is_none() {
            return Ok(());
        }
        let input = DimeNamedDigestV1 {
            name: name.take().context("trainer input name is absent")?,
            sha256: sha256.take().context("trainer input digest is absent")?,
        };
        inputs.push(input);
        Ok(())
    };
    let parse = |line: &str, key: &str| -> Option<String> {
        serde_json::from_str(line.trim().strip_prefix(&format!("{key} = "))?).ok()
    };
    let mut in_input = false;
    for line in body.lines() {
        if line.trim() == "[[input]]" {
            if in_input {
                flush(&mut name, &mut sha256, &mut inputs)?;
            }
            in_input = true;
        } else if in_input {
            if line.trim().starts_with('[') {
                flush(&mut name, &mut sha256, &mut inputs)?;
                in_input = false;
            } else if let Some(value) = parse(line, "name") {
                name = Some(value);
            } else if let Some(value) = parse(line, "sha256") {
                sha256 = Some(value);
            }
        }
    }
    if in_input {
        flush(&mut name, &mut sha256, &mut inputs)?;
    }
    inputs.sort_by(|left, right| left.name.cmp(&right.name));
    Ok(inputs)
}

pub(crate) fn trainer_scientific_payload(intent: &ActiveIntent) -> Result<DimeScientificPayloadV1> {
    let mut inputs = trainer_spec_inputs(&intent.spec_path)?;
    let binary_sha256 = inputs
        .iter()
        .find(|input| input.name == "script")
        .map(|input| input.sha256.clone())
        .context("trainer intent has no script input")?;
    let recipe = intent.trainer.recipe.as_ref().unwrap_or(&Value::Null);
    let configuration_sha256 = sha256_bytes(&serde_json::to_vec(&json!({
        "task": intent.trainer.task,
        "mode": intent.trainer.mode,
        "recipe": recipe,
    }))?);
    let parent_intent_sha256 = sha256_bytes(&serde_json::to_vec(&json!({
        "domain": "dime-trainer-scientific-intent-v1",
        "task": intent.trainer.task,
        "mode": intent.trainer.mode,
        "accepted_parent_checkpoint_sha256": intent.trainer.accepted_parent_checkpoint_sha256,
        "shadow_parent_checkpoint_sha256": intent.trainer.shadow_parent_checkpoint_sha256,
        "warm_start_checkpoint_sha256": intent.trainer.warm_start_checkpoint_sha256,
        "corpus_identity_sha256": intent.trainer.corpus_identity_sha256,
        "corpus_watermark": intent.trainer.corpus_watermark,
        "seed": intent.trainer.seed,
        "runtime_bundle_sha256": intent.trainer.runtime_bundle_sha256,
        "catalog_binding_sha256": intent.trainer.catalog_binding_sha256,
        "recipe": recipe,
    }))?);
    if inputs.is_empty() {
        inputs.push(DimeNamedDigestV1 {
            name: "legacy_spec".to_owned(),
            sha256: intent.trainer.immutable_spec_sha256.as_str().to_owned(),
        });
    }
    let payload = DimeScientificPayloadV1 {
        parent_intent_sha256,
        runtime_bundle_sha256: intent
            .trainer
            .runtime_bundle_sha256
            .as_ref()
            .map(|value| value.as_str().to_owned())
            .unwrap_or_else(|| digest_label("legacy-unbound-trainer-runtime")),
        binary_sha256,
        configuration_sha256,
        train_data_sha256: intent.trainer.corpus_identity_sha256.as_str().to_owned(),
        protocol_sha256: digest_label("jope-task-trainer-protocol-v3"),
        metric_sha256: digest_label("jope-conditional-pairwise-trainer-metric-v1"),
        seed: intent.trainer.seed,
        inputs,
    };
    payload.validate()?;
    Ok(payload)
}

pub(crate) fn trainer_initial_job_identity(
    current_job_identity: &str,
    root: &Path,
    payload: &DimeScientificPayloadV1,
) -> Result<String> {
    if let Some(first) = infrastructure_attempts(
        root,
        DimeRoleV1::AlternatingTrainer,
        &payload.identity_sha256()?,
    )?
    .first()
    {
        return Ok(first.job_identity.clone());
    }
    Ok(current_job_identity
        .rsplit_once("-infra")
        .filter(|(_, suffix)| {
            !suffix.is_empty() && suffix.bytes().all(|byte| byte.is_ascii_digit())
        })
        .map_or(current_job_identity, |(base, _)| base)
        .to_owned())
}

pub(crate) fn trainer_terminal_state<'a>(
    intent: &ActiveIntent,
    jobs: &'a [SchedulerJob],
    observed_at_ms: u128,
) -> Option<&'a str> {
    match jobs
        .iter()
        .find(|job| job.id == intent.trainer.immutable_job_identity)
    {
        Some(job)
            if matches!(
                job.state.as_str(),
                "failed" | "cancelled" | "preempted" | "lost" | "finished"
            ) =>
        {
            Some(job.state.as_str())
        }
        Some(_) => None,
        None if observed_at_ms.saturating_sub(intent.trainer.created_at_ms) >= 60_u128 * 1_000 => {
            Some("vanished")
        }
        None => None,
    }
}

pub(crate) fn trainer_scientific_failure_receipt(
    root: &Path,
    intent: &ActiveIntent,
    payload: &DimeScientificPayloadV1,
) -> Result<Option<String>> {
    let path = root
        .join("ops/dime-scientific-failures")
        .join(format!("{}.json", intent.trainer.immutable_job_identity));
    if !path.is_file() {
        return Ok(None);
    }
    let value = read_json(&path)?;
    let payload_sha256 = payload.identity_sha256()?;
    let evidence = value
        .get("evidence_sha256")
        .and_then(Value::as_str)
        .context("trainer scientific failure receipt lacks evidence")?;
    Sha256Digest::parse(evidence.to_owned())?;
    ensure!(
        value.get("schema").and_then(Value::as_str) == Some("DimeScientificFailure.v1")
            && value.get("job_identity").and_then(Value::as_str)
                == Some(intent.trainer.immutable_job_identity.as_str())
            && value
                .get("scientific_payload_sha256")
                .and_then(Value::as_str)
                == Some(payload_sha256.as_str())
            && value.get("deterministic").and_then(Value::as_bool) == Some(true),
        "trainer scientific failure receipt is not bound to the frozen intent"
    );
    Ok(Some(evidence.to_owned()))
}

pub(crate) struct TrainerRecoveryDisposition {
    pub(crate) action: &'static str,
    pub(crate) note: Option<String>,
}

pub(crate) fn recover_trainer_intent(
    args: &Args,
    state: &mut ControllerState,
    intent: &ActiveIntent,
    terminal_state: &str,
    observed_at_ms: u128,
) -> Result<TrainerRecoveryDisposition> {
    let payload = trainer_scientific_payload(intent)?;
    payload.validate()?;
    let initial_job_identity =
        trainer_initial_job_identity(&intent.trainer.immutable_job_identity, &args.root, &payload)?;
    let mut projection = load_or_rebuild_recovery_projection(
        &args.root,
        DimeRoleV1::AlternatingTrainer,
        &payload,
        &initial_job_identity,
    )?;
    if intent.trainer.immutable_job_identity != projection.current_job_identity {
        let next = reconcile_trainer_projection(args, intent, &projection, &payload)?;
        state.intent = Some(next);
        return Ok(TrainerRecoveryDisposition {
            action: "trainer_attempt_receipt_replayed",
            note: Some(
                "the existing append-only receipt restored its recorded next identity".to_owned(),
            ),
        });
    }
    if projection.scientific_failures >= 5 {
        return Ok(TrainerRecoveryDisposition {
            action: "trainer_scientific_failure_circuit_open",
            note: Some(
                "five hash-bound deterministic failures require scientific review".to_owned(),
            ),
        });
    }
    if let Some(drain) = matching_drain_intent(
        &args.root,
        DimeRoleV1::AlternatingTrainer,
        &payload,
        observed_at_ms,
    )? {
        append_event(
            &args.root,
            "trainer_intentional_drain_honored",
            &json!({
                "job_id": intent.trainer.immutable_job_identity,
                "drain_receipt_sha256": drain.content_sha256,
                "scientific_payload_sha256": payload.identity_sha256()?,
            }),
        )?;
        return Ok(TrainerRecoveryDisposition {
            action: "intentional_drain_honored",
            note: Some("hash-bound DIME drain suppresses infrastructure recovery".to_owned()),
        });
    }
    if projection.submitted_at_ms > 0 && observed_at_ms < projection.retry_not_before_ms {
        return Ok(TrainerRecoveryDisposition {
            action: "trainer_infrastructure_retry_backoff",
            note: Some(format!(
                "retry is bound to the frozen intent and due in {}s",
                projection
                    .retry_not_before_ms
                    .saturating_sub(observed_at_ms)
                    / 1_000
            )),
        });
    }
    let scientific = if terminal_state == "failed" {
        trainer_scientific_failure_receipt(&args.root, intent, &payload)?
    } else {
        None
    };
    let terminal_class = terminal_classification(terminal_state, scientific.as_deref())?;
    let scientific_failures = projection.scientific_failures.saturating_add(u32::from(
        terminal_class == DimeTerminalClassV1::ScientificFailure,
    ));
    let retry_allowed = scientific_failures < 5;
    let next_attempt = projection
        .current_attempt_number
        .checked_add(1)
        .context("trainer infrastructure attempt overflow")?;
    let current_job = &intent.trainer.immutable_job_identity;
    let base = current_job
        .rsplit_once("-infra")
        .filter(|(_, suffix)| suffix.bytes().all(|byte| byte.is_ascii_digit()))
        .map(|(base, _)| base)
        .unwrap_or(current_job);
    let next_job = format!("{base}-infra{next_attempt:06}");
    let evidence_sha256 = scientific.clone().unwrap_or_else(|| {
        sha256_bytes(
            format!(
                "dime-trainer-terminal-v1\0{}\0{}\0{}",
                current_job, terminal_state, intent.trainer.immutable_spec_sha256
            )
            .as_bytes(),
        )
    });
    let recent_cancellations = projection
        .foreign_cancellation_timestamps_ms
        .iter()
        .rev()
        .take_while(|timestamp| observed_at_ms.saturating_sub(**timestamp) <= 10 * 60 * 1_000)
        .count()
        + usize::from(terminal_state == "cancelled");
    let backoff_ms = if recent_cancellations >= 3 {
        15 * 60 * 1_000
    } else {
        jope_dime_regressor::dime_runtime_contracts::retry_backoff_ms(
            projection.current_attempt_number,
        )
    };
    let retry_not_before_ms = observed_at_ms.saturating_add(backoff_ms);
    let receipt = DimeInfrastructureAttemptV1::new_terminal(
        DimeRoleV1::AlternatingTrainer,
        payload.clone(),
        projection.current_attempt_number,
        current_job.clone(),
        retry_allowed.then_some(next_job.clone()),
        projection.last_attempt_receipt_sha256.clone(),
        intent.trainer.created_at_ms.max(1),
        retry_not_before_ms,
        DimeTerminalEvidenceV1 {
            terminal_state: terminal_state.to_owned(),
            class: terminal_class,
            evidence_sha256,
            observed_at_ms,
            foreign_cancellation: terminal_state == "cancelled",
        },
    )?;
    append_infrastructure_attempt(&args.root, &receipt)?;
    projection.apply_receipt(&receipt)?;
    let contention_alarm_open = projection.refresh_contention_alarm(observed_at_ms)?;
    save_recovery_projection(&args.root, &projection)?;
    if contention_alarm_open {
        append_event(
            &args.root,
            "trainer_foreign_cancellation_contention_alarm",
            &json!({
                "severity": "prominent",
                "job_id": current_job,
                "scientific_payload_sha256": projection.scientific_payload_sha256,
                "retry_mode": "capped_half_open",
                "retry_not_before_ms": projection.retry_not_before_ms,
            }),
        )?;
    }
    if !retry_allowed {
        append_event(
            &args.root,
            "trainer_scientific_failure_circuit_open",
            &json!({
                "job_id": current_job,
                "scientific_failures": projection.scientific_failures,
                "attempt_receipt_sha256": receipt.content_sha256,
            }),
        )?;
        return Ok(TrainerRecoveryDisposition {
            action: "trainer_scientific_failure_circuit_open",
            note: Some(
                "five hash-bound deterministic failures require scientific review".to_owned(),
            ),
        });
    }
    let next = reconcile_trainer_projection(args, intent, &projection, &payload)?;
    state.intent = Some(next.clone());
    state.updated_at_ms = observed_at_ms;
    save_state(
        &args.root.join("training/task-learning-controller.v2.json"),
        state,
    )?;
    let submission = "trainer_infrastructure_retry_audited";
    append_event(
        &args.root,
        if terminal_state == "cancelled" {
            "trainer_foreign_cancellation_recovered"
        } else {
            "trainer_infrastructure_attempt_recovered"
        },
        &json!({
            "prior_job_id": current_job,
            "next_job_id": next_job,
            "infrastructure_attempt": next_attempt,
            "scientific_payload_sha256": payload.identity_sha256()?,
            "attempt_receipt_sha256": receipt.content_sha256,
            "submission": submission,
            "retry_not_before_ms": projection.retry_not_before_ms,
            "contention_alarm_open": contention_alarm_open,
        }),
    )?;
    Ok(TrainerRecoveryDisposition {
        action: submission,
        note: Some("fresh job identity preserves corpus, parent, seed, recipe, and all staged input hashes".to_owned()),
    })
}

pub(crate) fn reconcile_trainer_projection(
    args: &Args,
    intent: &ActiveIntent,
    projection: &DimeRecoveryProjectionV1,
    payload: &DimeScientificPayloadV1,
) -> Result<ActiveIntent> {
    if intent.trainer.immutable_job_identity == projection.current_job_identity {
        return Ok(intent.clone());
    }
    ensure!(
        projection.retry_pending_submission,
        "trainer recovery projection changed identity without a pending retry"
    );
    let next_job = &projection.current_job_identity;
    let original = fs::read_to_string(&intent.spec_path)?;
    let replacement = format!("id = {}", serde_json::to_string(next_job)?);
    let mut replaced = false;
    let body = original
        .lines()
        .map(|line| {
            if !replaced && line.starts_with("id = ") {
                replaced = true;
                replacement.as_str()
            } else {
                line
            }
        })
        .collect::<Vec<_>>()
        .join("\n");
    ensure!(replaced, "trainer retry spec has no immutable job identity");
    let spec_path = args
        .root
        .join("specs/task-learning")
        .join(format!("{next_job}.toml"));
    write_immutable(&spec_path, body.as_bytes())?;
    let mut next = intent.clone();
    next.trainer.immutable_job_identity = next_job.clone();
    next.trainer.immutable_spec_sha256 = Sha256Digest::parse(sha256_bytes(body.as_bytes()))?;
    next.trainer.created_at_ms = projection.retry_not_before_ms.max(1);
    next.spec_path = spec_path;
    let observed_payload = trainer_scientific_payload(&next)?;
    ensure_same_scientific_payload(payload, &observed_payload)?;
    atomic_json(
        &args
            .root
            .join("training/trainer-intents")
            .join(format!("{next_job}.v3.json")),
        &serde_json::to_value(&next.trainer)?,
    )?;
    Ok(next)
}

pub(crate) fn shadow_canary_warm_start_eligible(
    root: &Path,
    task: Task,
    accepted_parent: &Sha256Digest,
    candidate: &Sha256Digest,
) -> Result<bool> {
    let directory = root.join("training/shadow-canaries/completed");
    if !directory.is_dir() {
        return Ok(false);
    }
    let mut matching = fs::read_dir(directory)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().extension().and_then(|value| value.to_str()) == Some("json"))
        .filter_map(|entry| {
            let value = read_json(&entry.path()).ok()?;
            let matches = value.get("schema").and_then(Value::as_str)
                == Some("ShadowBattleSummary.v1")
                && value.get("mode").and_then(Value::as_str) == Some("shadow")
                && value.get("task").and_then(Value::as_str) == Some(task.as_str())
                && value
                    .get("accepted_parent_checkpoint_sha256")
                    .and_then(Value::as_str)
                    == Some(accepted_parent.as_str())
                && value
                    .get("candidate_checkpoint_sha256")
                    .and_then(Value::as_str)
                    == Some(candidate.as_str());
            matches.then_some((
                value
                    .get("created_at_ms")
                    .and_then(Value::as_u64)
                    .unwrap_or(0),
                value,
            ))
        })
        .collect::<Vec<_>>();
    matching.sort_by_key(|(created_at_ms, _)| *created_at_ms);
    let Some((_, summary)) = matching.pop() else {
        return Ok(false);
    };
    Ok(shadow_summary_eligible(&summary))
}
