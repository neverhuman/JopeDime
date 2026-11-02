//! Task-aware alternating shadow/production trainer for any capable SCQ GPU lane.

use anyhow::{bail, ensure, Context, Result};
use clap::Parser;
use jope_dime_regressor::dime_runtime_contracts::{
    append_infrastructure_attempt, digest_label, ensure_same_scientific_payload,
    infrastructure_attempts, load_or_rebuild_recovery_projection, matching_drain_intent,
    save_recovery_projection, terminal_classification, DimeInfrastructureAttemptV1,
    DimeNamedDigestV1, DimeRecoveryProjectionV1, DimeRoleV1, DimeScientificPayloadV1,
    DimeTerminalClassV1, DimeTerminalEvidenceV1,
};
use jope_dime_regressor::paths;
use jope_dime_regressor::selfplay_runtime::{
    atomic_json, default_root_str, default_scq_str, now_ms, read_json, scq_json, sha256_bytes,
    sha256_file,
};
use jope_dime_regressor::selfplay_shadow::{
    deterministic_canary_contexts, validate_panel_consumption, CanaryContextV2,
    CausalEvidenceLedgerV3, PlateauCountedCandidateV1, PlateauDiagnosisV1, PlateauDiagnosisV2,
    PlateauFailureClassV2, PromotionContextScheduleV3, RecipeChangeDirectiveV1,
    RecipeChangeDirectiveV2, SealedPanelConsumptionV1, SealedPanelPoolV2, Sha256Digest,
    ShadowAlternationState, ShadowCheckpointV1, Task, TaskCanaryStreaks,
    TaskPromotionEvaluationIntentV3, TaskPromotionEvaluationV3, TrainerIntentV2, TrainerMode,
    PLATEAU_DIAGNOSIS_V2_SCHEMA, SEALED_PANEL_CONSUMPTION_V1_SCHEMA, SHADOW_CHECKPOINT_SCHEMA,
    TASK_PROMOTION_EVALUATION_INTENT_V3_SCHEMA, TRAINER_INTENT_V3_SCHEMA,
};
use jope_dime_regressor::spec_policy::TRAINER;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::thread;
use std::time::Duration;

const PROJECT: &str = "jope-selfplay-train";
const STATE_SCHEMA: &str = "TaskLearningControllerState.v2";
jope_paths::once_str!(
    fn default_runs_root_str => paths::scq_runs_root()
);
jope_paths::once_str!(
    fn default_trainer_str => paths::research_script("test_exact_train.py")
);
jope_paths::once_str!(
    fn default_common_str => paths::research_script("test_exact_common.py")
);
jope_paths::once_str!(
    fn default_packer_str => paths::research_script("test_exact_packed_corpus.py")
);
jope_paths::once_str!(
    fn default_prime_str => paths::runtime_bundle_pytorch()
);
const ACTIVE_STATES: &[&str] = &[
    "queued",
    "starting",
    "dispatching",
    "running",
    "preempting",
    "held",
];

fn main() -> Result<()> {
    let args = Args::parse();
    loop {
        match cycle(&args) {
            Ok(status) => {
                atomic_json(
                    &args.root.join("ops/task-learning-controller.v2.json"),
                    &status,
                )?;
                println!("{}", serde_json::to_string(&status)?);
            }
            Err(error) => {
                let failure = json!({
                    "schema": "TaskLearningControllerError.v2",
                    "generated_at_ms": now_ms(),
                    "error": format!("{error:#}"),
                });
                atomic_json(
                    &args.root.join("ops/task-learning-controller.v2.json"),
                    &failure,
                )?;
                eprintln!("{}", serde_json::to_string(&failure)?);
                if args.once {
                    return Err(error);
                }
            }
        }
        if args.once {
            return Ok(());
        }
        thread::sleep(Duration::from_secs(args.interval_seconds));
    }
}

fn cycle(args: &Args) -> Result<Value> {
    let state_path = args.root.join("training/task-learning-controller.v2.json");
    let mut state = load_state(&state_path)?;
    let mut accepted = accepted_pointers(&args.root)?;
    let promoted_tasks = apply_task_canary_acceptances(args, &mut state, &accepted)?;
    if !promoted_tasks.is_empty() {
        accepted = accepted_pointers(&args.root)?;
    }
    state.updated_at_ms = now_ms();
    save_state(&state_path, &state)?;
    reset_changed_shadow_epochs(&mut state, &accepted);
    let corpus = corpus_pointer(&args.root)?;
    // The recipe each task's shadow lane would train with RIGHT NOW; plateau
    // scoping compares rejected shadows against this. A task whose current
    // recipe cannot be computed scopes conservatively (v1 behavior).
    let mut current_recipe_identities = BTreeMap::new();
    for task in Task::ALL {
        let Some(pointer) = accepted.get(&task) else {
            continue;
        };
        let Ok((_, recipe)) = trainer_recipe(
            &corpus,
            task,
            TrainerMode::Shadow,
            &pointer.digest,
            state
                .last_recipe_e_corpus_watermark
                .get(&task)
                .copied()
                .unwrap_or(0),
            state
                .last_recipe_e_descendant_receipts
                .get(&task)
                .copied()
                .unwrap_or(0),
        ) else {
            continue;
        };
        if let Ok(identity) = recipe_identity_sha256(&recipe) {
            current_recipe_identities.insert(task, identity);
        }
    }
    let plateau_tasks =
        publish_plateau_diagnoses(&args.root, &accepted, &current_recipe_identities)?;
    let accepted_digests = accepted
        .iter()
        .map(|(task, pointer)| (*task, pointer.digest.clone()))
        .collect::<BTreeMap<_, _>>();
    let jobs = scheduler_jobs(&scq_json(&args.scq, &["ls"])?)?;

    if let Some(mut intent) = state.intent.clone() {
        let payload = trainer_scientific_payload(&intent)?;
        let initial_job_identity = trainer_initial_job_identity(
            &intent.trainer.immutable_job_identity,
            &args.root,
            &payload,
        )?;
        let mut recovery = load_or_rebuild_recovery_projection(
            &args.root,
            DimeRoleV1::AlternatingTrainer,
            &payload,
            &initial_job_identity,
        )?;
        if intent.trainer.immutable_job_identity != recovery.current_job_identity {
            intent = reconcile_trainer_projection(args, &intent, &recovery, &payload)?;
            state.intent = Some(intent.clone());
            state.updated_at_ms = now_ms();
            save_state(&state_path, &state)?;
            return Ok(status(
                args,
                &state,
                &accepted,
                &corpus,
                "trainer_attempt_receipt_replayed",
                Some("the append-only attempt receipt restored its recorded next identity"),
            ));
        }
        if recovery.retry_pending_submission {
            if now_ms() < recovery.retry_not_before_ms {
                return Ok(status(
                    args,
                    &state,
                    &accepted,
                    &corpus,
                    "trainer_infrastructure_retry_backoff",
                    Some("the receipt-bound retry is waiting for its recorded backoff"),
                ));
            }
            let action = if args.submit {
                let action = submit_or_register(
                    args,
                    &intent.spec_path,
                    &intent.trainer.immutable_job_identity,
                )?;
                recovery.mark_submitted(&intent.trainer.immutable_job_identity, now_ms())?;
                save_recovery_projection(&args.root, &recovery)?;
                action
            } else {
                "trainer_infrastructure_retry_audited"
            };
            return Ok(status(
                args,
                &state,
                &accepted,
                &corpus,
                action,
                Some("retry remains bound to the same scientific payload"),
            ));
        }
        if jobs.iter().any(|job| {
            job.id == intent.trainer.immutable_job_identity
                && job.project == PROJECT
                && ACTIVE_STATES.contains(&job.state.as_str())
        }) {
            return Ok(status(
                args,
                &state,
                &accepted,
                &corpus,
                "active_trainer_preserved",
                None,
            ));
        }
        if let Some(result) = find_train_result(
            &args.runs_root,
            &intent.trainer.immutable_job_identity,
            &intent.output_name,
        )? {
            harvest_result(args, &mut state, &intent, &result)?;
            state.intent = None;
            state.updated_at_ms = now_ms();
            save_state(&state_path, &state)?;
        } else if admission_request_pending(&args.root, &intent.trainer.immutable_job_identity)? {
            return Ok(status(
                args,
                &state,
                &accepted,
                &corpus,
                "waiting_for_gpu_admission",
                None,
            ));
        } else if let Some(terminal_state) = trainer_terminal_state(&intent, &jobs, now_ms()) {
            let disposition =
                recover_trainer_intent(args, &mut state, &intent, terminal_state, now_ms())?;
            state.updated_at_ms = now_ms();
            save_state(&state_path, &state)?;
            return Ok(status(
                args,
                &state,
                &accepted,
                &corpus,
                disposition.action,
                disposition.note.as_deref(),
            ));
        } else {
            return Ok(status(
                args,
                &state,
                &accepted,
                &corpus,
                "trainer_result_pending",
                Some("frozen trainer intent is within the scheduler appearance grace period"),
            ));
        }
    }

    let active_trainers = jobs
        .iter()
        .filter(|job| {
            job.project == PROJECT
                && ACTIVE_STATES.contains(&job.state.as_str())
                && is_task_trainer_job_id(&job.id)
        })
        .map(|job| job.id.clone())
        .collect::<Vec<_>>();
    if !active_trainers.is_empty() {
        return Ok(status(
            args,
            &state,
            &accepted,
            &corpus,
            "foreign_or_legacy_trainer_preserved",
            Some("the task-aware controller never preempts a running trainer"),
        ));
    }

    let ledger = load_ledger(&args.root)?;
    let (task, mode) = if let Some(task) = ledger
        .as_ref()
        .and_then(|ledger| production_task_v3(ledger, &accepted))
    {
        (task, TrainerMode::Production)
    } else {
        let mut selected = None;
        for _ in 0..2 {
            let candidate = state.alternation.next(&accepted_digests);
            if plateau_tasks.contains(&candidate) {
                continue;
            }
            let trained = state
                .last_shadow_corpus_watermark
                .get(&candidate)
                .copied()
                .unwrap_or(0);
            if corpus.watermark > trained {
                selected = Some(candidate);
                break;
            }
        }
        let Some(task) = selected else {
            state.updated_at_ms = now_ms();
            save_state(&state_path, &state)?;
            let plateau = !plateau_tasks.is_empty();
            return Ok(status(
                args,
                &state,
                &accepted,
                &corpus,
                if plateau {
                    "plateau_diagnosis_requires_recipe_change"
                } else {
                    "waiting_for_newer_corpus"
                },
                plateau.then_some(
                    "three distinct shadows were rejected on an unchanged parent; identical shadow training is paused",
                ),
            ));
        };
        (task, TrainerMode::Shadow)
    };

    let accepted_pointer = accepted.get(&task).context("task pointer is absent")?;
    let shadow_parent = if mode == TrainerMode::Shadow {
        match state.shadow_models.get(&task) {
            Some(model)
                if model.accepted_parent_checkpoint_sha256 == accepted_pointer.digest
                    && model.checkpoint_path.is_file()
                    && shadow_canary_warm_start_eligible(
                        &args.root,
                        task,
                        &accepted_pointer.digest,
                        &model.checkpoint_sha256,
                    )? =>
            {
                Some(model)
            }
            _ => None,
        }
    } else {
        None
    };
    let (warm_start_path, warm_start_digest, shadow_parent_digest) =
        if let Some(parent) = shadow_parent {
            (
                parent.checkpoint_path.clone(),
                parent.checkpoint_sha256.clone(),
                Some(parent.checkpoint_sha256.clone()),
            )
        } else {
            (
                accepted_pointer.path.clone(),
                accepted_pointer.digest.clone(),
                None,
            )
        };
    ensure!(
        sha256_file(&warm_start_path)? == warm_start_digest.as_str(),
        "task trainer warm-start content drift"
    );
    let sequence = state.sequence;
    let seed = args.seed_base.saturating_add(sequence);
    let mode_name = match mode {
        TrainerMode::Shadow => "shadow",
        TrainerMode::Production => "production",
    };
    let job_id = format!(
        "jope-selfplay-{mode_name}-{}-{:09}-s{:06}",
        task.as_str(),
        corpus.watermark,
        sequence
    );
    let output_name = format!(
        "{mode_name}-{}-{:09}-s{:06}",
        task.as_str(),
        corpus.watermark,
        sequence
    );
    let spec_path = args
        .root
        .join("specs/task-learning")
        .join(format!("{job_id}.toml"));
    let (recipe_cli, recipe) = trainer_recipe(
        &corpus,
        task,
        mode,
        &accepted_pointer.digest,
        state
            .last_recipe_e_corpus_watermark
            .get(&task)
            .copied()
            .unwrap_or(0),
        state
            .last_recipe_e_descendant_receipts
            .get(&task)
            .copied()
            .unwrap_or(0),
    )?;
    let spec = render_spec(
        args,
        &job_id,
        &output_name,
        task,
        mode,
        seed,
        &corpus,
        &warm_start_path,
        &warm_start_digest,
        &accepted_pointer.digest,
        recipe_cli,
    )?;
    write_immutable(&spec_path, spec.as_bytes())?;
    let (runtime_bundle_path, runtime_bundle_sha256, catalog_binding_path, catalog_binding_sha256) =
        trainer_v3_pins(&args.root, &corpus)?;
    let intent = TrainerIntentV2 {
        schema: TRAINER_INTENT_V3_SCHEMA.to_owned(),
        task,
        mode,
        accepted_parent_checkpoint_sha256: accepted_pointer.digest.clone(),
        shadow_parent_checkpoint_sha256: shadow_parent_digest,
        warm_start_checkpoint_sha256: warm_start_digest,
        corpus_identity_sha256: Sha256Digest::parse(corpus.packed_manifest_sha256.clone())?,
        corpus_watermark: corpus.watermark,
        seed,
        immutable_job_identity: job_id.clone(),
        immutable_spec_sha256: Sha256Digest::parse(sha256_bytes(spec.as_bytes()))?,
        created_at_ms: now_ms(),
        runtime_bundle_path: Some(runtime_bundle_path),
        runtime_bundle_sha256: Some(runtime_bundle_sha256),
        catalog_binding_path: Some(catalog_binding_path),
        catalog_binding_sha256: Some(catalog_binding_sha256),
        recipe: Some(recipe),
    };
    intent.validate()?;
    let active_intent = ActiveIntent {
        trainer: intent.clone(),
        spec_path: spec_path.clone(),
        output_name,
        warm_start_path,
    };
    let scientific_payload = trainer_scientific_payload(&active_intent)?;
    let mut recovery = load_or_rebuild_recovery_projection(
        &args.root,
        DimeRoleV1::AlternatingTrainer,
        &scientific_payload,
        &job_id,
    )?;
    state.intent = Some(active_intent);
    state.sequence = state.sequence.saturating_add(1);
    state.updated_at_ms = now_ms();
    save_state(&state_path, &state)?;
    atomic_json(
        &args
            .root
            .join("training/trainer-intents")
            .join(format!("{job_id}.v3.json")),
        &serde_json::to_value(&intent)?,
    )?;
    append_event(&args.root, "trainer_intent_created", &intent)?;

    let action = if args.submit {
        let action = submit_or_register(args, &spec_path, &job_id)?;
        recovery.record_initial_submission(&job_id, now_ms())?;
        save_recovery_projection(&args.root, &recovery)?;
        action
    } else {
        "audit_intent_only"
    };
    Ok(status(args, &state, &accepted, &corpus, action, None))
}

mod execution;
mod recipe;
mod scheduler;
mod state;
mod trainer;

use execution::*;
use recipe::*;
use scheduler::*;
use state::*;
use trainer::*;

#[cfg(test)]
mod tests;
