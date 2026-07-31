//! Task-aware alternating shadow/production trainer for any capable SCQ GPU lane.

use anyhow::{bail, ensure, Context, Result};
use clap::Parser;
use jope_dime_regressor::selfplay_runtime::{
    atomic_json, now_ms, read_json, scq_json, sha256_bytes, sha256_file, DEFAULT_ROOT, DEFAULT_SCQ,
};
use jope_dime_regressor::selfplay_shadow::{
    deterministic_canary_contexts, validate_panel_consumption, CanaryContextV2,
    CausalEvidenceLedgerV3, PlateauDiagnosisV1, SealedPanelConsumptionV1, SealedPanelPoolV2,
    Sha256Digest, ShadowAlternationState, ShadowCheckpointV1, Task, TaskCanaryStreaks,
    TaskPromotionEvaluationV3, TrainerIntentV2, TrainerMode, SEALED_PANEL_CONSUMPTION_V1_SCHEMA,
    SHADOW_CHECKPOINT_SCHEMA, TRAINER_INTENT_V3_SCHEMA,
};
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
const DEFAULT_RUNS_ROOT: &str = "/home/ubuntu/smart_cluster/var/runs";
const DEFAULT_TRAINER: &str = "/home/ubuntu/jope-dime/scripts/test_exact_train.py";
const DEFAULT_COMMON: &str = "/home/ubuntu/jope-dime/scripts/test_exact_common.py";
const DEFAULT_PACKER: &str = "/home/ubuntu/jope-dime/scripts/test_exact_packed_corpus.py";
const DEFAULT_PRIME: &str = "/home/ubuntu/jope-prime/src";
const ACTIVE_STATES: &[&str] = &[
    "queued",
    "starting",
    "dispatching",
    "running",
    "preempting",
    "held",
];

#[derive(Parser, Debug)]
#[command(
    name = "jope-task-learning-controller",
    about = "Alternate quarantined task trainers and preempt with strictly gated production"
)]
struct Args {
    #[arg(long, default_value = DEFAULT_ROOT)]
    root: PathBuf,
    #[arg(long, default_value = DEFAULT_SCQ)]
    scq: PathBuf,
    #[arg(long, default_value = DEFAULT_RUNS_ROOT)]
    runs_root: PathBuf,
    #[arg(long, default_value = DEFAULT_TRAINER)]
    trainer_script: PathBuf,
    #[arg(long, default_value = DEFAULT_COMMON)]
    common_script: PathBuf,
    #[arg(long, default_value = DEFAULT_PACKER)]
    packed_script: PathBuf,
    #[arg(long, default_value = DEFAULT_PRIME)]
    prime_source: PathBuf,
    #[arg(long)]
    submit: bool,
    #[arg(long)]
    once: bool,
    #[arg(long, default_value_t = 30, value_parser = clap::value_parser!(u64).range(1..))]
    interval_seconds: u64,
    #[arg(long, default_value_t = 20_260_729)]
    seed_base: u64,
}

#[derive(Clone, Debug, Deserialize)]
struct SchedulerJob {
    id: String,
    project: String,
    state: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
struct ModelRef {
    accepted_parent_checkpoint_sha256: Sha256Digest,
    checkpoint_sha256: Sha256Digest,
    checkpoint_path: PathBuf,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct ActiveIntent {
    trainer: TrainerIntentV2,
    spec_path: PathBuf,
    output_name: String,
    warm_start_path: PathBuf,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
struct ControllerState {
    schema: String,
    sequence: u64,
    alternation: ShadowAlternationState,
    shadow_models: BTreeMap<Task, ModelRef>,
    last_shadow_corpus_watermark: BTreeMap<Task, u64>,
    last_recipe_e_corpus_watermark: BTreeMap<Task, u64>,
    last_recipe_e_descendant_receipts: BTreeMap<Task, u64>,
    production_candidates: BTreeMap<Task, ModelRef>,
    canary_streaks: TaskCanaryStreaks,
    processed_canary_receipts: BTreeSet<String>,
    intent: Option<ActiveIntent>,
    updated_at_ms: u128,
}

impl Default for ControllerState {
    fn default() -> Self {
        Self {
            schema: STATE_SCHEMA.to_owned(),
            sequence: 1,
            alternation: ShadowAlternationState::default(),
            shadow_models: BTreeMap::new(),
            last_shadow_corpus_watermark: BTreeMap::new(),
            last_recipe_e_corpus_watermark: BTreeMap::new(),
            last_recipe_e_descendant_receipts: BTreeMap::new(),
            production_candidates: BTreeMap::new(),
            canary_streaks: TaskCanaryStreaks::default(),
            processed_canary_receipts: BTreeSet::new(),
            intent: None,
            updated_at_ms: now_ms(),
        }
    }
}

#[derive(Clone, Debug)]
struct AcceptedPointer {
    digest: Sha256Digest,
    path: PathBuf,
}

#[derive(Clone, Debug)]
struct CorpusPointer {
    watermark: u64,
    manifest: PathBuf,
    manifest_sha256: String,
    packed_manifest: PathBuf,
    packed_manifest_sha256: String,
}

fn trainer_v3_pins(
    root: &Path,
    corpus: &CorpusPointer,
) -> Result<(PathBuf, Sha256Digest, PathBuf, Sha256Digest)> {
    let executable = fs::canonicalize(std::env::current_exe()?)?;
    let runtime_bundle = executable
        .parent()
        .and_then(Path::parent)
        .and_then(Path::parent)
        .context("task controller is not running from a runtime bundle")?
        .to_path_buf();
    let runtime_sha = Sha256Digest::parse(
        runtime_bundle
            .file_name()
            .and_then(|value| value.to_str())
            .context("runtime bundle lacks a content identity")?
            .to_owned(),
    )?;
    let manifest = read_json(&corpus.manifest)?;
    if let (Some(path), Some(digest)) = (
        manifest.get("catalog_binding").and_then(Value::as_str),
        manifest
            .get("catalog_binding_sha256")
            .and_then(Value::as_str),
    ) {
        let path = PathBuf::from(path);
        ensure!(
            path.is_file() && sha256_file(&path)? == digest,
            "trainer catalog binding content drift"
        );
        return Ok((
            runtime_bundle,
            runtime_sha,
            path,
            Sha256Digest::parse(digest.to_owned())?,
        ));
    }
    let pointer_path = root.join("catalog/current.json");
    let pointer = read_json(&pointer_path)?;
    ensure!(
        manifest.get("catalog_sha256").and_then(Value::as_str)
            == pointer.get("catalog_sha256").and_then(Value::as_str),
        "legacy corpus has no CatalogBinding.v1 and the live catalog advanced"
    );
    let catalog = PathBuf::from(
        pointer
            .get("catalog")
            .and_then(Value::as_str)
            .context("catalog pointer lacks catalog")?,
    );
    let data = PathBuf::from(
        pointer
            .get("data")
            .and_then(Value::as_str)
            .context("catalog pointer lacks data")?,
    );
    let pointer_snapshot = root
        .join("training/catalog-bindings/pointer-snapshots")
        .join(format!("{}.json", sha256_file(&pointer_path)?));
    atomic_json(&pointer_snapshot, &pointer)?;
    let created_at_ms = pointer_path
        .metadata()?
        .modified()?
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis();
    let binding = json!({
        "schema": "CatalogBinding.v1",
        "catalog_path": catalog,
        "catalog_sha256": pointer.get("catalog_sha256"),
        "data_path": data,
        "data_sha256": pointer.get("data_sha256"),
        "pointer_snapshot_path": pointer_snapshot,
        "pointer_snapshot_sha256": sha256_file(&pointer_snapshot)?,
        "created_at_ms": created_at_ms,
    });
    let identity = sha256_bytes(&serde_json::to_vec(&binding)?);
    let binding_path = root
        .join("training/catalog-bindings")
        .join(format!("{identity}.json"));
    atomic_json(&binding_path, &binding)?;
    let binding_sha = Sha256Digest::parse(sha256_file(&binding_path)?)?;
    Ok((runtime_bundle, runtime_sha, binding_path, binding_sha))
}

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
        state.updated_at_ms = now_ms();
        save_state(&state_path, &state)?;
    }
    reset_changed_shadow_epochs(&mut state, &accepted);
    let plateau_tasks = publish_plateau_diagnoses(&args.root, &accepted)?;
    let accepted_digests = accepted
        .iter()
        .map(|(task, pointer)| (*task, pointer.digest.clone()))
        .collect::<BTreeMap<_, _>>();
    let corpus = corpus_pointer(&args.root)?;
    let jobs = scheduler_jobs(&scq_json(&args.scq, &["ls"])?)?;

    if let Some(intent) = state.intent.clone() {
        if audited_terminal_retry(&args.root, &intent, &jobs)? {
            state.intent = None;
            state.alternation.next_task = Some(intent.trainer.task);
            state.updated_at_ms = now_ms();
            save_state(&state_path, &state)?;
            append_event(
                &args.root,
                "trainer_terminal_failure_audited",
                &json!({
                    "job_id": intent.trainer.immutable_job_identity,
                    "task": intent.trainer.task,
                    "retry_from_accepted": true,
                }),
            )?;
        } else if jobs.iter().any(|job| {
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
        } else {
            return Ok(status(
                args,
                &state,
                &accepted,
                &corpus,
                "trainer_result_pending_or_failed",
                Some("intent remains immutable until an output or operator-audited terminal failure exists"),
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
    state.intent = Some(ActiveIntent {
        trainer: intent.clone(),
        spec_path: spec_path.clone(),
        output_name,
        warm_start_path,
    });
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
        submit_or_register(args, &spec_path, &job_id)?
    } else {
        "audit_intent_only"
    };
    Ok(status(args, &state, &accepted, &corpus, action, None))
}

fn is_task_trainer_job_id(job_id: &str) -> bool {
    [
        "jope-selfplay-shadow-",
        "jope-selfplay-production-",
        "jope-selfplay-trainer-",
    ]
    .iter()
    .any(|prefix| job_id.starts_with(prefix))
}

fn audited_terminal_retry(
    root: &Path,
    intent: &ActiveIntent,
    jobs: &[SchedulerJob],
) -> Result<bool> {
    let terminal = jobs.iter().any(|job| {
        job.id == intent.trainer.immutable_job_identity
            && matches!(job.state.as_str(), "failed" | "cancelled")
    });
    if !terminal {
        return Ok(false);
    }
    let path = root
        .join("ops/trainer-recovery-audits")
        .join(format!("{}.json", intent.trainer.immutable_job_identity));
    if !path.is_file() {
        return Ok(false);
    }
    let audit = read_json(&path)?;
    ensure!(
        audit.get("schema").and_then(Value::as_str) == Some("TrainerRecoveryAudit.v1")
            && audit.get("job_id").and_then(Value::as_str)
                == Some(intent.trainer.immutable_job_identity.as_str())
            && audit.get("immutable_spec_sha256").and_then(Value::as_str)
                == Some(intent.trainer.immutable_spec_sha256.as_str())
            && audit.get("disposition").and_then(Value::as_str) == Some("retry_from_accepted"),
        "trainer recovery audit is not bound to the immutable failed intent"
    );
    Ok(true)
}

fn shadow_canary_warm_start_eligible(
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

fn shadow_summary_eligible(summary: &Value) -> bool {
    let metric = summary.get("candidate_vs_accepted").unwrap_or(&Value::Null);
    summary
        .get("contexts")
        .and_then(Value::as_array)
        .is_some_and(|contexts| contexts.len() == 16)
        && metric
            .get("validity")
            .and_then(Value::as_f64)
            .is_some_and(|value| value >= 0.99)
        && summary
            .get("accepted_validity")
            .and_then(Value::as_f64)
            .is_some_and(|value| value >= 0.99)
        && summary
            .get("gp_validity")
            .and_then(Value::as_f64)
            .is_some_and(|value| value >= 0.99)
        && metric
            .get("pairwise_lift")
            .and_then(Value::as_f64)
            .is_some_and(|value| value >= -0.005)
        && metric
            .get("latency_ratio")
            .and_then(Value::as_f64)
            .is_some_and(|value| value <= 1.25)
        && metric
            .get("stability")
            .and_then(Value::as_f64)
            .is_some_and(|value| value >= 0.99)
        && summary
            .get("replay_verification_complete")
            .and_then(Value::as_bool)
            == Some(true)
}

fn publish_plateau_diagnoses(
    root: &Path,
    accepted: &BTreeMap<Task, AcceptedPointer>,
) -> Result<BTreeSet<Task>> {
    let directory = root.join("training/shadow-canaries/completed");
    let mut plateau_tasks = BTreeSet::new();
    if !directory.is_dir() {
        return Ok(plateau_tasks);
    }
    for task in Task::ALL {
        let parent = match accepted.get(&task) {
            Some(pointer) => &pointer.digest,
            // Legacy tasks must always have an accepted pointer; the new
            // tasks are skipped until their first anchor is promoted.
            None if matches!(task, Task::Binary | Task::Regression) => {
                bail!("accepted pointer lacks {task}")
            }
            None => continue,
        };
        let mut by_candidate = BTreeMap::<Sha256Digest, (u64, Value)>::new();
        for entry in fs::read_dir(&directory)?.filter_map(|entry| entry.ok()) {
            let Ok(value) = read_json(&entry.path()) else {
                continue;
            };
            if value.get("schema").and_then(Value::as_str) != Some("ShadowBattleSummary.v1")
                || value.get("mode").and_then(Value::as_str) != Some("shadow")
                || value.get("task").and_then(Value::as_str) != Some(task.as_str())
                || value
                    .get("accepted_parent_checkpoint_sha256")
                    .and_then(Value::as_str)
                    != Some(parent.as_str())
            {
                continue;
            }
            let Some(candidate) = value
                .get("candidate_checkpoint_sha256")
                .and_then(Value::as_str)
                .and_then(|digest| Sha256Digest::parse(digest.to_owned()).ok())
            else {
                continue;
            };
            let created = value
                .get("created_at_ms")
                .and_then(Value::as_u64)
                .unwrap_or_default();
            let replace = by_candidate
                .get(&candidate)
                .is_none_or(|(known, _)| created >= *known);
            if replace {
                by_candidate.insert(candidate, (created, value));
            }
        }
        let mut rejected = by_candidate
            .into_iter()
            .filter(|(_, (_, summary))| !shadow_summary_eligible(summary))
            .collect::<Vec<_>>();
        rejected.sort_by_key(|(_, (created, _))| *created);
        if rejected.len() < 3 {
            continue;
        }
        let selected = &rejected[rejected.len() - 3..];
        let proposal_validity = selected
            .iter()
            .filter_map(|(_, (_, summary))| {
                summary.pointer("/candidate_vs_accepted/validity")?.as_f64()
            })
            .fold(1.0, f64::min);
        let gp_disadvantage = selected
            .iter()
            .filter_map(|(_, (_, summary))| {
                summary.pointer("/candidate_vs_gp/pairwise_lift")?.as_f64()
            })
            .map(|lift| (-lift).max(0.0))
            .fold(0.0, f64::max);
        let latency = selected
            .iter()
            .filter_map(|(_, (_, summary))| {
                summary
                    .pointer("/candidate_vs_accepted/latency_ratio")?
                    .as_f64()
            })
            .fold(0.0, f64::max);
        let replayed = selected
            .iter()
            .filter(|(_, (_, summary))| {
                summary
                    .get("replay_verification_complete")
                    .and_then(Value::as_bool)
                    == Some(true)
            })
            .count();
        let diagnosis = PlateauDiagnosisV1 {
            task,
            unchanged_parent_checkpoint_sha256: parent.clone(),
            rejected_candidates: selected
                .iter()
                .map(|(candidate, _)| candidate.clone())
                .collect(),
            proposal_validity,
            gp_disadvantage,
            calibration_summary:
                "learned uncertainty/risk ranking remains disabled until its family-disjoint calibration gate passes"
                    .to_owned(),
            data_slice_summary:
                "the recycled 16-context diagnostic has no authoritative provenance-v2 slice result; single-use external panels are required"
                    .to_owned(),
            latency_summary: format!("worst candidate/accepted latency ratio {latency:.6}"),
            replay_summary: format!("{replayed}/3 rejected shadow receipts replay-verified"),
            diagnosed_at_ms: now_ms(),
        };
        diagnosis.validate()?;
        let path = root.join("training/plateau-diagnoses").join(format!(
            "{}-{}.v1.json",
            task.as_str(),
            parent.as_str()
        ));
        if path.is_file() {
            let existing: PlateauDiagnosisV1 = serde_json::from_value(read_json(&path)?)?;
            existing.validate()?;
            ensure!(
                existing.unchanged_parent_checkpoint_sha256 == *parent,
                "plateau diagnosis parent drift"
            );
        } else {
            atomic_json(&path, &serde_json::to_value(&diagnosis)?)?;
        }
        plateau_tasks.insert(task);
    }
    Ok(plateau_tasks)
}

fn load_state(path: &Path) -> Result<ControllerState> {
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

fn save_state(path: &Path, state: &ControllerState) -> Result<()> {
    ensure!(
        state.schema == STATE_SCHEMA,
        "task learning state schema drift"
    );
    atomic_json(path, &serde_json::to_value(state)?)
}

fn accepted_pointers(root: &Path) -> Result<BTreeMap<Task, AcceptedPointer>> {
    let value = read_json(&root.join("training/accepted-by-task.json"))?;
    ensure!(
        value.get("schema").and_then(Value::as_str) == Some("JopeSelfplayAcceptedByTask.v1"),
        "accepted-by-task pointer schema drift"
    );
    let mut output = BTreeMap::new();
    for task in Task::ALL {
        let row = match value.get("tasks").and_then(|tasks| tasks.get(task.as_str())) {
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

fn reset_changed_shadow_epochs(
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

fn apply_task_canary_acceptances(
    args: &Args,
    state: &mut ControllerState,
    accepted: &BTreeMap<Task, AcceptedPointer>,
) -> Result<Vec<Task>> {
    let directory = args
        .root
        .join("training/task-promotion-evaluations/accepted");
    if !directory.is_dir() {
        return Ok(Vec::new());
    }
    let mut receipts = fs::read_dir(&directory)?
        .filter_map(|entry| entry.ok())
        .filter(|entry| entry.path().extension().and_then(|value| value.to_str()) == Some("json"))
        .filter_map(|entry| {
            let value = read_json(&entry.path()).ok()?;
            let evaluation: TaskPromotionEvaluationV3 = serde_json::from_value(value).ok()?;
            Some((evaluation.evaluated_at_ms, entry.path(), evaluation))
        })
        .collect::<Vec<_>>();
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
            .context("accepted task pointer is absent")?;
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
        if ready && !promoted.contains(&task) {
            promote_task_pointer(args, task, parent, &candidate, &independent_passes)?;
            state.canary_streaks.reset_after_promotion(task);
            state.production_candidates.remove(&task);
            state.shadow_models.remove(&task);
            state.last_shadow_corpus_watermark.remove(&task);
            state.last_recipe_e_corpus_watermark.remove(&task);
            state.last_recipe_e_descendant_receipts.remove(&task);
            promoted.push(task);
        }
    }
    Ok(promoted)
}

fn independent_external_passes(
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
    if rows.len() < 3 {
        return Ok(Vec::new());
    }
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
            == 3,
        "external promotion runs must use disjoint seeds"
    );
    Ok(receipt_ids)
}

fn promote_task_pointer(
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
    ensure!(
        pointer
            .get("tasks")
            .and_then(|tasks| tasks.get(task.as_str()))
            .and_then(|row| row.get("checkpoint_sha256"))
            .and_then(Value::as_str)
            == Some(parent.digest.as_str()),
        "accepted task pointer changed before compare-and-swap"
    );
    let row = pointer
        .get_mut("tasks")
        .and_then(Value::as_object_mut)
        .and_then(|tasks| tasks.get_mut(task.as_str()))
        .and_then(Value::as_object_mut)
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

fn corpus_pointer(root: &Path) -> Result<CorpusPointer> {
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

fn load_ledger(root: &Path) -> Result<Option<CausalEvidenceLedgerV3>> {
    let path = root.join("training/causal-evidence-ledger.v3.json");
    if !path.is_file() {
        return Ok(None);
    }
    let ledger: CausalEvidenceLedgerV3 = serde_json::from_value(read_json(&path)?)?;
    ledger.validate()?;
    Ok(Some(ledger))
}

fn production_task_v3(
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

fn trainer_recipe(
    corpus: &CorpusPointer,
    task: Task,
    mode: TrainerMode,
    accepted_parent: &Sha256Digest,
    last_recipe_e_watermark: u64,
    last_recipe_e_descendant_receipts: u64,
) -> Result<(&'static str, Value)> {
    let broad = || {
        json!({
            "recipe_name": "a",
            "broad_replay_fraction": 1.0,
            "exact_causal_pair_fraction": 0.0,
            "hard_novel_oversampling": false,
            "fixed_anchor_holdout": true,
        })
    };
    if mode != TrainerMode::Shadow {
        return Ok(("a", broad()));
    }
    let packed = read_json(&corpus.packed_manifest)?;
    let Some(relative) = packed.get("causal_pair_index").and_then(Value::as_str) else {
        return Ok(("a", broad()));
    };
    let path = corpus
        .packed_manifest
        .parent()
        .context("packed manifest has no parent")?
        .join(relative);
    let expected = packed
        .get("files")
        .and_then(|files| files.get(relative))
        .and_then(|row| row.get("sha256"))
        .and_then(Value::as_str)
        .context("packed causal pair index lacks a content pin")?;
    ensure!(
        path.is_file() && sha256_file(&path)? == expected,
        "packed causal pair index content drift"
    );
    let mut complete_parent_pairs = 0_usize;
    for line in BufReader::new(File::open(&path)?).lines() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let row: Value = serde_json::from_str(&line)?;
        if row.get("task").and_then(Value::as_str) == Some(task.as_str())
            && row
                .get("accepted_parent_checkpoint_sha256")
                .and_then(Value::as_str)
                == Some(accepted_parent.as_str())
        {
            complete_parent_pairs += 1;
        }
    }
    if complete_parent_pairs < 32 {
        return Ok(("a", broad()));
    }
    if let Some(relative) = packed
        .get("recipe_e_evidence_index")
        .and_then(Value::as_str)
    {
        let path = corpus
            .packed_manifest
            .parent()
            .context("packed manifest has no parent")?
            .join(relative);
        let expected = packed
            .get("files")
            .and_then(|files| files.get(relative))
            .and_then(|row| row.get("sha256"))
            .and_then(Value::as_str)
            .context("packed recipe E evidence index lacks a content pin")?;
        ensure!(
            path.is_file() && sha256_file(&path)? == expected,
            "packed recipe E evidence index content drift"
        );
        let mut completed_descendant_rows = 0_u64;
        let mut completed_descendant_receipts = BTreeSet::new();
        let mut hard_negative_rows = 0_u64;
        for line in BufReader::new(File::open(&path)?).lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let row: Value = serde_json::from_str(&line)?;
            if row.get("task").and_then(Value::as_str) != Some(task.as_str()) {
                continue;
            }
            match row.get("lane").and_then(Value::as_str) {
                Some("completed_descendant_trajectories") => {
                    let receipt = row
                        .get("descendant_credit_receipt_sha256")
                        .and_then(Value::as_str)
                        .context("recipe E descendant row lacks receipt identity")?;
                    Sha256Digest::parse(receipt.to_owned())?;
                    ensure!(
                        matches!(
                            row.get("descendant_horizon").and_then(Value::as_u64),
                            Some(1 | 2 | 4 | 8)
                        ),
                        "recipe E descendant row has an invalid horizon"
                    );
                    completed_descendant_rows += 1;
                    completed_descendant_receipts.insert(receipt.to_owned());
                }
                Some("validity_instability_hard_negatives") => {
                    hard_negative_rows += 1;
                }
                _ => bail!("packed recipe E evidence lane is invalid"),
            }
        }
        let completed_descendant_receipts = u64::try_from(completed_descendant_receipts.len())?;
        let new_eligible_labels = corpus.watermark.saturating_sub(last_recipe_e_watermark);
        let new_completed_descendant_receipts =
            completed_descendant_receipts.saturating_sub(last_recipe_e_descendant_receipts);
        let trigger_open =
            new_eligible_labels >= 50_000 || new_completed_descendant_receipts >= 5_000;
        if trigger_open && completed_descendant_rows > 0 && hard_negative_rows > 0 {
            return Ok((
                "e",
                json!({
                    "recipe_name": "e",
                    "uniform_broad_replay_fraction": 0.50,
                    "exact_current_parent_pair_fraction": 0.25,
                    "completed_descendant_trajectory_fraction": 0.15,
                    "validity_instability_hard_negative_fraction": 0.10,
                    "complete_parent_pairs": complete_parent_pairs,
                    "completed_descendant_rows": completed_descendant_rows,
                    "completed_descendant_receipts": completed_descendant_receipts,
                    "hard_negative_rows": hard_negative_rows,
                    "new_eligible_labels": new_eligible_labels,
                    "new_completed_descendant_receipts":
                        new_completed_descendant_receipts,
                    "accepted_parent_checkpoint_sha256": accepted_parent,
                    "shadow_only": true,
                    "fixed_anchor_holdout": true,
                    "test_adaptive": true,
                    "benchmark_valid": false,
                }),
            ));
        }
    }
    Ok((
        "d",
        json!({
            "recipe_name": "d",
            "broad_replay_fraction": 0.75,
            "exact_causal_pair_fraction": 0.25,
            "hard_novel_oversampling": true,
            "fixed_anchor_holdout": true,
            "complete_parent_pairs": complete_parent_pairs,
            "accepted_parent_checkpoint_sha256": accepted_parent,
        }),
    ))
}

#[allow(clippy::too_many_arguments)]
fn render_spec(
    args: &Args,
    job_id: &str,
    output_name: &str,
    task: Task,
    mode: TrainerMode,
    seed: u64,
    corpus: &CorpusPointer,
    warm_path: &Path,
    warm_digest: &Sha256Digest,
    accepted_parent: &Sha256Digest,
    recipe: &str,
) -> Result<String> {
    let packed: Value = read_json(&corpus.packed_manifest)?;
    let packed_root = corpus
        .packed_manifest
        .parent()
        .context("packed manifest has no parent")?;
    let catalog = PathBuf::from(
        packed
            .get("catalog_data")
            .and_then(Value::as_str)
            .context("packed manifest lacks catalog data")?,
    );
    let catalog_sha = packed
        .get("catalog_data_sha256")
        .and_then(Value::as_str)
        .context("packed manifest lacks catalog digest")?;
    ensure!(
        catalog.is_file() && sha256_file(&catalog)? == catalog_sha,
        "packed catalog content drift"
    );
    let mode_name = match mode {
        TrainerMode::Shadow => "shadow",
        TrainerMode::Production => "production",
    };
    let quoted = |value: &str| serde_json::to_string(value).expect("JSON string");
    let mut text = format!(
        "[job]\nid = {}\nproject = \"{PROJECT}\"\ntier = \"normal\"\n\
         command = \"/usr/bin/python3\"\n\
         args = [\"{{input:script}}\", \"train\", \"--corpus-manifest\", \"{{input:manifest}}\", \
         \"--output\", \"{{output:training}}/{output_name}\", \"--device\", \"cuda\", \
         \"--warm\", \"{{input:checkpoint}}\", \"--recipe\", \"{recipe}\", \"--seed\", \"{seed}\", \
         \"--task\", \"{}\", \"--mode\", \"{mode_name}\", \"--packed-manifest\", \
         \"{{input:packed_manifest}}\", \"--packed-catalog-data\", \"{{input:packed_catalog}}\", \
         \"--accepted-parent-sha256\", \"{}\"]\n\n\
         [resource]\ngpus_per_task = 1\ncpus_reserved = 4\nomp_threads = 4\n\
         ram_mb = 16384\ngang_size = 1\nplacement = \"single_node\"\n\n\
         [[dependencies.smoke]]\nname = \"trainer GPU capability\"\n\
         command = \"/usr/bin/nvidia-smi\"\n\
         args = [\"--query-gpu=uuid,memory.total\", \"--format=csv,noheader\"]\n\
         timeout_secs = 5\n\n\
         [[dependencies.smoke]]\nname = \"trainer Python package\"\n\
         command = \"/usr/bin/env\"\nargs = [{}, \"/usr/bin/python3\", \"-c\", \"import jope\"]\n\
         timeout_secs = 30\n\n\
         [env]\nPYTHONPATH = {}\nOMP_NUM_THREADS = \"4\"\n\
         JOPE_TRAINER_INTENT_SCHEMA = \"TrainerIntent.v3\"\n\n\
         [preempt]\npreemptible = true\ncheckpoint = \"none\"\nresume = \"rerun\"\n\n\
         [limits]\nmax_runtime_hours = 168\nmax_restarts = 1\n\n",
        quoted(job_id),
        task.as_str(),
        accepted_parent.as_str(),
        quoted(&format!(
            "PYTHONPATH={}",
            args.prime_source.to_string_lossy()
        )),
        quoted(&args.prime_source.to_string_lossy()),
    );
    let mut inputs = vec![
        (
            "script",
            args.trainer_script.clone(),
            "scripts/test_exact_train.py".to_owned(),
            None,
        ),
        (
            "common",
            args.common_script.clone(),
            "scripts/test_exact_common.py".to_owned(),
            None,
        ),
        (
            "packed_source",
            args.packed_script.clone(),
            "scripts/test_exact_packed_corpus.py".to_owned(),
            None,
        ),
        (
            "manifest",
            corpus.manifest.clone(),
            "corpus/corpus-manifest.json".to_owned(),
            Some(corpus.manifest_sha256.clone()),
        ),
        (
            "checkpoint",
            warm_path.to_owned(),
            format!("checkpoint/{}.model.pt", warm_digest.as_str()),
            Some(warm_digest.as_str().to_owned()),
        ),
        (
            "packed_manifest",
            corpus.packed_manifest.clone(),
            "packed/manifest.json".to_owned(),
            Some(corpus.packed_manifest_sha256.clone()),
        ),
        (
            "packed_catalog",
            catalog,
            "packed/catalog-data.f32le".to_owned(),
            Some(catalog_sha.to_owned()),
        ),
        (
            "trainer_source",
            args.prime_source.join("jope/policy/train_v3.py"),
            "provenance/train_v3.py".to_owned(),
            None,
        ),
    ];
    let files = packed
        .get("files")
        .and_then(Value::as_object)
        .context("packed manifest lacks files")?;
    for (index, (name, metadata)) in files.iter().enumerate() {
        let path = packed_root.join(name);
        let digest = metadata
            .get("sha256")
            .and_then(Value::as_str)
            .context("packed file lacks digest")?;
        inputs.push((
            Box::leak(format!("packed_{index}").into_boxed_str()),
            path,
            format!("packed/{name}"),
            Some(digest.to_owned()),
        ));
    }
    for (name, source, target, pinned) in inputs {
        ensure!(
            source.is_file(),
            "trainer input is absent: {}",
            source.display()
        );
        let digest = pinned.unwrap_or(sha256_file(&source)?);
        ensure!(
            sha256_file(&source)? == digest,
            "trainer input content drift: {}",
            source.display()
        );
        text.push_str(&format!(
            "[[input]]\nname = {}\nsource = {}\ntarget = {}\nsha256 = {}\n\n",
            quoted(name),
            quoted(&source.to_string_lossy()),
            quoted(&target),
            quoted(&digest),
        ));
    }
    text.push_str("[[output]]\nname = \"training\"\nsource = \"training\"\ndirectory = true\n");
    Ok(text)
}

fn submit_or_register(args: &Args, spec: &Path, job_id: &str) -> Result<&'static str> {
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

fn register_admission(args: &Args, spec: &Path, job_id: &str) -> Result<()> {
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

fn harvest_result(
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

fn write_production_canary_intents(
    args: &Args,
    intent: &ActiveIntent,
    candidate: &Sha256Digest,
) -> Result<()> {
    let pool: SealedPanelPoolV2 = serde_json::from_value(read_json(
        &args.root.join("training/sealed-panel-pool.v2.json"),
    )?)?;
    pool.validate()?;
    let consumed_directory = args.root.join("training/sealed-panels/consumed");
    let mut prior = if consumed_directory.is_dir() {
        fs::read_dir(&consumed_directory)?
            .filter_map(|entry| entry.ok())
            .filter_map(|entry| {
                serde_json::from_value::<SealedPanelConsumptionV1>(read_json(&entry.path()).ok()?)
                    .ok()
            })
            .collect::<Vec<_>>()
    } else {
        Vec::new()
    };
    let consumed_panels = prior
        .iter()
        .map(|receipt| receipt.panel.as_str())
        .collect::<BTreeSet<_>>();
    let mut panels = pool
        .entries
        .iter()
        .filter(|entry| entry.task == intent.trainer.task)
        .filter_map(|entry| entry.promotion_panel.as_deref())
        .filter(|panel| !consumed_panels.contains(panel))
        .collect::<BTreeSet<_>>()
        .into_iter()
        .filter(|panel| pool.promotion_panel(intent.trainer.task, panel).len() >= 128)
        .take(3)
        .collect::<Vec<_>>();
    panels.sort_unstable();
    ensure!(
        panels.len() == 3,
        "sealed pool cannot supply three unused task-local promotion panels"
    );
    for run_index in 0..3_usize {
        let panel = panels[run_index];
        let available = pool.promotion_panel(intent.trainer.task, panel);
        let scheduled = (available.len().min(512) - 128) / 64 * 64 + 128;
        let contexts = available
            .into_iter()
            .take(scheduled)
            .map(|context| context.context_sha256.clone())
            .collect::<Vec<_>>();
        let consumption = SealedPanelConsumptionV1 {
            schema: SEALED_PANEL_CONSUMPTION_V1_SCHEMA.to_owned(),
            task: intent.trainer.task,
            panel: panel.to_owned(),
            candidate_checkpoint_sha256: candidate.clone(),
            run_index,
            seed: intent.trainer.seed.saturating_add(run_index as u64 + 1),
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
        prior.push(consumption);
        let rows = contexts
            .into_iter()
            .map(|context_sha256| {
                json!({
                    "context_sha256": context_sha256,
                })
            })
            .collect::<Vec<_>>();
        atomic_json(
            &args
                .root
                .join("training/task-promotion-evaluations/pending")
                .join(format!(
                    "{}-{}-run{}.v3.json",
                    intent.trainer.task.as_str(),
                    candidate.as_str(),
                    run_index
                )),
            &json!({
                "schema": "TaskPromotionEvaluationIntent.v3",
                "task": intent.trainer.task,
                "accepted_parent_checkpoint_sha256":
                    intent.trainer.accepted_parent_checkpoint_sha256,
                "candidate_checkpoint_sha256": candidate,
                "run_index": run_index,
                "seed": intent.trainer.seed.saturating_add(run_index as u64 + 1),
                "panel_consumption": consumption_path,
                "panel_consumption_sha256": consumption_sha,
                "context_schedule": rows,
                "initial_contexts": 128,
                "context_step": 64,
                "maximum_contexts": scheduled,
                "bootstrap_resamples": 10_000,
                "created_at_ms": now_ms(),
            }),
        )?;
    }
    Ok(())
}

fn write_shadow_canary_intent(args: &Args, checkpoint: &ShadowCheckpointV1) -> Result<()> {
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

fn select_canary_contexts(
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

fn find_train_result(runs_root: &Path, job_id: &str, output_name: &str) -> Result<Option<PathBuf>> {
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

fn admission_request_pending(root: &Path, job_id: &str) -> Result<bool> {
    let path = root.join("ops/gpu-lane-admission/requests/alternating_trainer.json");
    if !path.is_file() {
        return Ok(false);
    }
    Ok(read_json(&path)?
        .get("immutable_job_identity")
        .and_then(Value::as_str)
        == Some(job_id))
}

fn scheduler_jobs(value: &Value) -> Result<Vec<SchedulerJob>> {
    Ok(serde_json::from_value(
        value
            .get("Jobs")
            .and_then(|value| value.get("jobs"))
            .cloned()
            .context("SCQ response lacks Jobs.jobs")?,
    )?)
}

fn command_message(output: &std::process::Output) -> String {
    String::from_utf8_lossy(if output.stderr.is_empty() {
        &output.stdout
    } else {
        &output.stderr
    })
    .trim()
    .to_owned()
}

fn capacity_backoff(message: &str) -> bool {
    message.contains("dependency-unsatisfied")
        && (message.contains("no eligible launch-ready node currently")
            || message.contains("eligible_node launch-ready-capacity")
            || (message.contains("trainer GPU capability") && message.contains("exit 1")))
}

fn write_immutable(path: &Path, bytes: &[u8]) -> Result<()> {
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

fn copy_immutable(source: &Path, target: &Path, digest: &str) -> Result<()> {
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

fn append_event<T: Serialize>(root: &Path, kind: &str, detail: &T) -> Result<()> {
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

fn status(
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

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[test]
    fn causal_recipe_requires_32_exact_pairs_and_is_shadow_only() {
        let directory = tempdir().unwrap();
        let packed_root = directory.path().join("packed");
        fs::create_dir_all(&packed_root).unwrap();
        let pair_index = packed_root.join("causal_pairs.jsonl");
        let accepted = Sha256Digest::parse("a".repeat(64)).unwrap();
        let rows = (0..32)
            .map(|index| {
                serde_json::to_string(&json!({
                    "pair_id": format!("{index:064x}"),
                    "task": "binary",
                    "accepted_parent_checkpoint_sha256": accepted,
                }))
                .unwrap()
            })
            .collect::<Vec<_>>();
        fs::write(&pair_index, format!("{}\n", rows[..31].join("\n"))).unwrap();
        let packed_manifest = packed_root.join("manifest.json");
        let write_manifest = || {
            atomic_json(
                &packed_manifest,
                &json!({
                    "causal_pair_index": "causal_pairs.jsonl",
                    "files": {
                        "causal_pairs.jsonl": {
                            "sha256": sha256_file(&pair_index).unwrap(),
                        }
                    }
                }),
            )
            .unwrap();
        };
        write_manifest();
        let corpus = CorpusPointer {
            watermark: 50_000,
            manifest: directory.path().join("unused-corpus.json"),
            manifest_sha256: String::new(),
            packed_manifest: packed_manifest.clone(),
            packed_manifest_sha256: String::new(),
        };

        let (recipe, _) =
            trainer_recipe(&corpus, Task::Binary, TrainerMode::Shadow, &accepted, 0, 0).unwrap();
        assert_eq!(recipe, "a");

        fs::write(&pair_index, format!("{}\n", rows.join("\n"))).unwrap();
        write_manifest();
        let (recipe, metadata) =
            trainer_recipe(&corpus, Task::Binary, TrainerMode::Shadow, &accepted, 0, 0).unwrap();
        assert_eq!(recipe, "d");
        assert_eq!(metadata["complete_parent_pairs"], 32);
        assert_eq!(metadata["broad_replay_fraction"], 0.75);
        assert_eq!(metadata["exact_causal_pair_fraction"], 0.25);

        let (recipe, metadata) = trainer_recipe(
            &corpus,
            Task::Binary,
            TrainerMode::Production,
            &accepted,
            0,
            0,
        )
        .unwrap();
        assert_eq!(recipe, "a");
        assert_eq!(metadata["exact_causal_pair_fraction"], 0.0);

        let evidence_index = packed_root.join("recipe_e_evidence.jsonl");
        fs::write(
            &evidence_index,
            format!(
                "{}\n{}\n",
                serde_json::to_string(&json!({
                    "ordinal": 0,
                    "task": "binary",
                    "lane": "completed_descendant_trajectories",
                    "descendant_credit_receipt_sha256": "b".repeat(64),
                    "descendant_horizon": 4,
                }))
                .unwrap(),
                serde_json::to_string(&json!({
                    "ordinal": 1,
                    "task": "binary",
                    "lane": "validity_instability_hard_negatives",
                }))
                .unwrap(),
            ),
        )
        .unwrap();
        atomic_json(
            &packed_manifest,
            &json!({
                "causal_pair_index": "causal_pairs.jsonl",
                "recipe_e_evidence_index": "recipe_e_evidence.jsonl",
                "files": {
                    "causal_pairs.jsonl": {
                        "sha256": sha256_file(&pair_index).unwrap(),
                    },
                    "recipe_e_evidence.jsonl": {
                        "sha256": sha256_file(&evidence_index).unwrap(),
                    }
                }
            }),
        )
        .unwrap();
        let (recipe, metadata) =
            trainer_recipe(&corpus, Task::Binary, TrainerMode::Shadow, &accepted, 0, 0).unwrap();
        assert_eq!(recipe, "e");
        assert_eq!(metadata["completed_descendant_receipts"], 1);
        assert_eq!(metadata["hard_negative_rows"], 1);
        assert_eq!(metadata["benchmark_valid"], false);

        let (recipe, _) = trainer_recipe(
            &corpus,
            Task::Binary,
            TrainerMode::Shadow,
            &accepted,
            49_999,
            1,
        )
        .unwrap();
        assert_eq!(recipe, "d");
    }

    #[test]
    fn changed_task_parent_resets_only_that_shadow_lineage() {
        let directory = tempdir().unwrap();
        let model = directory.path().join("shadow.model.pt");
        fs::write(&model, b"model").unwrap();
        let old = Sha256Digest::parse("a".repeat(64)).unwrap();
        let changed = Sha256Digest::parse("b".repeat(64)).unwrap();
        let stable = Sha256Digest::parse("c".repeat(64)).unwrap();
        let mut state = ControllerState::default();
        state.shadow_models.insert(
            Task::Regression,
            ModelRef {
                accepted_parent_checkpoint_sha256: old,
                checkpoint_sha256: changed.clone(),
                checkpoint_path: model.clone(),
            },
        );
        state.shadow_models.insert(
            Task::Binary,
            ModelRef {
                accepted_parent_checkpoint_sha256: stable.clone(),
                checkpoint_sha256: changed.clone(),
                checkpoint_path: model,
            },
        );
        let accepted = BTreeMap::from([
            (
                Task::Regression,
                AcceptedPointer {
                    digest: changed,
                    path: PathBuf::new(),
                },
            ),
            (
                Task::Binary,
                AcceptedPointer {
                    digest: stable,
                    path: PathBuf::new(),
                },
            ),
        ]);
        reset_changed_shadow_epochs(&mut state, &accepted);
        assert!(!state.shadow_models.contains_key(&Task::Regression));
        assert!(state.shadow_models.contains_key(&Task::Binary));
    }

    #[test]
    fn promotion_compare_and_swap_mutates_only_one_task_pointer() {
        let directory = tempdir().unwrap();
        let root = directory.path();
        fs::create_dir_all(root.join("training")).unwrap();
        let parent_path = root.join("parent.model.pt");
        let child_path = root.join("child.model.pt");
        fs::write(&parent_path, b"parent").unwrap();
        fs::write(&child_path, b"child").unwrap();
        let parent_digest = Sha256Digest::parse(sha256_file(&parent_path).unwrap()).unwrap();
        let child_digest = Sha256Digest::parse(sha256_file(&child_path).unwrap()).unwrap();
        atomic_json(
            &root.join("training/accepted-by-task.json"),
            &json!({
                "schema": "JopeSelfplayAcceptedByTask.v1",
                "global_checkpoint_sha256": parent_digest,
                "global_pointer": null,
                "tasks": {
                    "binary": {
                        "checkpoint": parent_path,
                        "checkpoint_sha256": parent_digest,
                        "promotion_tier": null,
                        "source": "test",
                        "updated_at_ms": 1
                    },
                    "regression": {
                        "checkpoint": parent_path,
                        "checkpoint_sha256": parent_digest,
                        "promotion_tier": null,
                        "source": "test",
                        "updated_at_ms": 1
                    }
                },
                "updated_at_ms": 1
            }),
        )
        .unwrap();
        let args = Args {
            root: root.to_owned(),
            scq: PathBuf::from("/bin/false"),
            runs_root: root.join("runs"),
            trainer_script: root.join("trainer"),
            common_script: root.join("common"),
            packed_script: root.join("packed"),
            prime_source: root.join("prime"),
            submit: false,
            once: true,
            interval_seconds: 1,
            seed_base: 1,
        };
        promote_task_pointer(
            &args,
            Task::Regression,
            &AcceptedPointer {
                digest: parent_digest.clone(),
                path: parent_path,
            },
            &ModelRef {
                accepted_parent_checkpoint_sha256: parent_digest.clone(),
                checkpoint_sha256: child_digest.clone(),
                checkpoint_path: child_path,
            },
            &[
                "receipt-1".to_owned(),
                "receipt-2".to_owned(),
                "receipt-3".to_owned(),
            ],
        )
        .unwrap();
        let pointer = read_json(&root.join("training/accepted-by-task.json")).unwrap();
        assert_eq!(
            pointer["tasks"]["regression"]["checkpoint_sha256"],
            child_digest.as_str()
        );
        assert_eq!(
            pointer["tasks"]["binary"]["checkpoint_sha256"],
            parent_digest.as_str()
        );
        assert!(pointer["global_checkpoint_sha256"].is_null());
    }

    #[test]
    fn proposer_jobs_do_not_block_the_independent_trainer_lane() {
        assert!(is_task_trainer_job_id(
            "jope-selfplay-shadow-binary-005263844-s000006"
        ));
        assert!(is_task_trainer_job_id(
            "jope-selfplay-production-regression-005263844-s000007"
        ));
        assert!(is_task_trainer_job_id("jope-selfplay-trainer-legacy"));
        assert!(!is_task_trainer_job_id(
            "jope-selfplay-cohort-propose-fbcd5459f2c1-wave"
        ));
        assert!(!is_task_trainer_job_id(
            "jope-canary-propose-dc0072851563-447531cc"
        ));
    }

    #[test]
    fn shadow_warm_start_requires_a_complete_passing_canary() {
        let directory = tempdir().unwrap();
        let root = directory.path();
        let completed = root.join("training/shadow-canaries/completed");
        fs::create_dir_all(&completed).unwrap();
        let parent = Sha256Digest::parse("a".repeat(64)).unwrap();
        let candidate = Sha256Digest::parse("b".repeat(64)).unwrap();
        let path = completed.join("summary.json");
        let summary = |validity: f64| {
            json!({
                "schema": "ShadowBattleSummary.v1",
                "mode": "shadow",
                "task": "binary",
                "accepted_parent_checkpoint_sha256": parent,
                "candidate_checkpoint_sha256": candidate,
                "created_at_ms": 1,
                "contexts": vec![json!({"context": "x"}); 16],
                "accepted_validity": 1.0,
                "gp_validity": 1.0,
                "replay_verification_complete": true,
                "candidate_vs_accepted": {
                    "validity": validity,
                    "pairwise_lift": 0.0,
                    "latency_ratio": 1.0,
                    "stability": 1.0
                }
            })
        };
        atomic_json(&path, &summary(1.0)).unwrap();
        assert!(
            shadow_canary_warm_start_eligible(root, Task::Binary, &parent, &candidate).unwrap()
        );
        atomic_json(&path, &summary(0.9375)).unwrap();
        assert!(
            !shadow_canary_warm_start_eligible(root, Task::Binary, &parent, &candidate).unwrap()
        );
    }

    #[test]
    fn three_rejected_shadows_publish_and_hold_a_parent_plateau() {
        let directory = tempdir().unwrap();
        let root = directory.path();
        let completed = root.join("training/shadow-canaries/completed");
        fs::create_dir_all(&completed).unwrap();
        let parent = Sha256Digest::parse("a".repeat(64)).unwrap();
        for (index, character) in ["1", "2", "3"].into_iter().enumerate() {
            let candidate = character.repeat(64);
            atomic_json(
                &completed.join(format!("{candidate}.json")),
                &json!({
                    "schema": "ShadowBattleSummary.v1",
                    "mode": "shadow",
                    "task": "binary",
                    "accepted_parent_checkpoint_sha256": parent,
                    "candidate_checkpoint_sha256": candidate,
                    "created_at_ms": index + 1,
                    "contexts": vec![json!({"context": "x"}); 16],
                    "accepted_validity": 1.0,
                    "gp_validity": 1.0,
                    "replay_verification_complete": true,
                    "candidate_vs_accepted": {
                        "validity": 0.98,
                        "pairwise_lift": -0.01,
                        "latency_ratio": 1.1,
                        "stability": 1.0
                    },
                    "candidate_vs_gp": {
                        "pairwise_lift": -0.04
                    }
                }),
            )
            .unwrap();
        }
        let accepted = BTreeMap::from([
            (
                Task::Binary,
                AcceptedPointer {
                    digest: parent.clone(),
                    path: PathBuf::new(),
                },
            ),
            (
                Task::Regression,
                AcceptedPointer {
                    digest: Sha256Digest::parse("b".repeat(64)).unwrap(),
                    path: PathBuf::new(),
                },
            ),
        ]);
        let plateau = publish_plateau_diagnoses(root, &accepted).unwrap();
        assert_eq!(plateau, BTreeSet::from([Task::Binary]));
        let path = root
            .join("training/plateau-diagnoses")
            .join(format!("binary-{}.v1.json", parent.as_str()));
        let diagnosis: PlateauDiagnosisV1 =
            serde_json::from_value(read_json(&path).unwrap()).unwrap();
        assert_eq!(diagnosis.rejected_candidates.len(), 3);
        assert_eq!(diagnosis.gp_disadvantage, 0.04);
    }
}
