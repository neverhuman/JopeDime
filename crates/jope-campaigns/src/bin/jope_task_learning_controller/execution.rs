use super::*;

mod recipe;
pub(super) use recipe::trainer_recipe;

pub(super) fn production_task_v3(
    ledger: &CausalEvidenceLedgerV3,
    accepted: &BTreeMap<Task, AcceptedPointer>,
) -> Option<Task> {
    Task::ALL.into_iter().find(|task| {
        ledger.promotion_epoch.get(task).is_some_and(|row| {
            row.production_gate_open
                && accepted.get(task).is_some_and(|pointer| {
                    row.evidence.accepted_parent_checkpoint_sha256.as_ref() == Some(&pointer.digest)
                })
        })
    })
}

mod render;
pub(super) use render::render_spec;

pub(super) fn submit_or_register(args: &Args, spec: &Path, job_id: &str) -> Result<&'static str> {
    let check = Command::new(&args.scq).arg("check").arg(spec).output()?;
    let check_message = command_message(&check);
    if !check.status.success() {
        if capacity_backoff(&check_message) {
            register_admission(args, spec, job_id)?;
            return Ok("gpu_admission_requested");
        }
        bail!("task trainer dependency verification failed: {check_message}");
    }
    let submit = Command::new(&args.scq).arg("submit").arg(spec).output()?;
    let message = command_message(&submit);
    if submit.status.success() || message.to_ascii_lowercase().contains("already exists") {
        return Ok("trainer_submitted");
    }
    if capacity_backoff(&message) {
        register_admission(args, spec, job_id)?;
        return Ok("gpu_admission_requested");
    }
    bail!("task trainer submission failed: {message}")
}

pub(super) fn register_admission(args: &Args, spec: &Path, job_id: &str) -> Result<()> {
    let executable = fs::canonicalize(std::env::current_exe()?)?;
    let controller = executable
        .parent()
        .context("task controller executable has no bin directory")?
        .join("jope_gpu_lane_admission");
    let output = Command::new(controller)
        .args([
            "--root",
            args.root.to_string_lossy().as_ref(),
            "--register-role",
            "alternating-trainer",
            "--spec",
            spec.to_string_lossy().as_ref(),
            "--job-id",
            job_id,
        ])
        .output()?;
    if !output.status.success() {
        bail!(
            "task trainer admission registration failed: {}",
            String::from_utf8_lossy(&output.stderr).trim()
        );
    }
    Ok(())
}

pub(super) fn harvest_result(
    args: &Args,
    state: &mut ControllerState,
    intent: &ActiveIntent,
    result_path: &Path,
) -> Result<()> {
    let result = read_json(result_path)?;
    ensure!(
        result.get("task").and_then(Value::as_str) == Some(intent.trainer.task.as_str()),
        "trainer result task mismatch"
    );
    let expected_mode = match intent.trainer.mode {
        TrainerMode::Shadow => "shadow",
        TrainerMode::Production => "production",
    };
    ensure!(
        result.get("trainer_mode").and_then(Value::as_str) == Some(expected_mode),
        "trainer result mode mismatch"
    );
    ensure!(
        result
            .get("accepted_ancestor_sha256")
            .and_then(Value::as_str)
            == Some(intent.trainer.warm_start_checkpoint_sha256.as_str()),
        "trainer result warm-start lineage mismatch"
    );
    let digest = Sha256Digest::parse(
        result
            .get("checkpoint_sha256")
            .and_then(Value::as_str)
            .context("trainer result lacks checkpoint digest")?
            .to_owned(),
    )?;
    let source = result_path
        .parent()
        .context("trainer result has no parent")?
        .join(format!("{}.model.pt", digest.as_str()));
    ensure!(
        source.is_file() && sha256_file(&source)? == digest.as_str(),
        "trainer result checkpoint artifact drift"
    );
    let directory = match intent.trainer.mode {
        TrainerMode::Shadow => args
            .root
            .join("training/shadow-checkpoints")
            .join(intent.trainer.task.as_str()),
        TrainerMode::Production => args
            .root
            .join("training/production-candidates")
            .join(intent.trainer.task.as_str()),
    };
    fs::create_dir_all(&directory)?;
    let target = directory.join(format!("{}.model.pt", digest.as_str()));
    copy_immutable(&source, &target, digest.as_str())?;
    let model = ModelRef {
        accepted_parent_checkpoint_sha256: intent.trainer.accepted_parent_checkpoint_sha256.clone(),
        checkpoint_sha256: digest.clone(),
        checkpoint_path: target.clone(),
    };
    match intent.trainer.mode {
        TrainerMode::Shadow => {
            let checkpoint = ShadowCheckpointV1 {
                schema: SHADOW_CHECKPOINT_SCHEMA.to_owned(),
                task: intent.trainer.task,
                accepted_parent_checkpoint_sha256: intent
                    .trainer
                    .accepted_parent_checkpoint_sha256
                    .clone(),
                shadow_parent_checkpoint_sha256: intent
                    .trainer
                    .shadow_parent_checkpoint_sha256
                    .clone(),
                checkpoint_sha256: digest.clone(),
                corpus_identity_sha256: intent.trainer.corpus_identity_sha256.clone(),
                corpus_watermark: intent.trainer.corpus_watermark,
                model_hashes: BTreeMap::from([("policy".to_owned(), digest.clone())]),
                trainer_intent_sha256: Sha256Digest::of_bytes(&serde_json::to_vec(
                    &intent.trainer,
                )?),
                created_at_ms: now_ms(),
                promotion_eligible: false,
            };
            checkpoint.validate()?;
            atomic_json(
                &directory.join(format!("{}.v1.json", digest.as_str())),
                &serde_json::to_value(&checkpoint)?,
            )?;
            write_shadow_canary_intent(args, &checkpoint)?;
            state.shadow_models.insert(intent.trainer.task, model);
            state
                .alternation
                .last_shadow_by_task
                .insert(intent.trainer.task, digest);
            state
                .last_shadow_corpus_watermark
                .insert(intent.trainer.task, intent.trainer.corpus_watermark);
            if intent
                .trainer
                .recipe
                .as_ref()
                .and_then(|recipe| recipe.get("recipe_name"))
                .and_then(Value::as_str)
                == Some("e")
            {
                let descendant_receipts = intent
                    .trainer
                    .recipe
                    .as_ref()
                    .and_then(|recipe| recipe.get("completed_descendant_receipts"))
                    .and_then(Value::as_u64)
                    .context("recipe E intent lacks descendant receipt count")?;
                state
                    .last_recipe_e_corpus_watermark
                    .insert(intent.trainer.task, intent.trainer.corpus_watermark);
                state
                    .last_recipe_e_descendant_receipts
                    .insert(intent.trainer.task, descendant_receipts);
            }
            append_event(&args.root, "shadow_checkpoint_quarantined", &checkpoint)?;
        }
        TrainerMode::Production => {
            atomic_json(
                &directory.join(format!("{}.candidate.v1.json", digest.as_str())),
                &json!({
                    "schema": "ProductionCheckpointCandidate.v1",
                    "task": intent.trainer.task,
                    "accepted_parent_checkpoint_sha256":
                        intent.trainer.accepted_parent_checkpoint_sha256,
                    "checkpoint_sha256": digest,
                    "checkpoint": target,
                    "trainer_intent": intent.trainer,
                    "promotion_requires_independent_task_canary_acceptances": 3,
                    "created_at_ms": now_ms(),
                }),
            )?;
            state
                .production_candidates
                .insert(intent.trainer.task, model);
            write_production_canary_intents(args, intent, &digest)?;
        }
    }
    Ok(())
}
