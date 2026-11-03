use super::*;

#[derive(Parser, Debug)]
#[command(
    name = "jope-task-learning-controller",
    about = "Alternate quarantined task trainers and preempt with strictly gated production"
)]
pub(super) struct Args {
    #[arg(long, default_value = default_root_str())]
    pub(super) root: PathBuf,
    #[arg(long, default_value = default_scq_str())]
    pub(super) scq: PathBuf,
    #[arg(long, default_value = default_runs_root_str())]
    pub(super) runs_root: PathBuf,
    #[arg(long, default_value = default_trainer_str())]
    pub(super) trainer_script: PathBuf,
    #[arg(long, default_value = default_common_str())]
    pub(super) common_script: PathBuf,
    #[arg(long, default_value = default_packer_str())]
    pub(super) packed_script: PathBuf,
    #[arg(long, default_value = default_prime_str())]
    pub(super) prime_source: PathBuf,
    #[arg(long)]
    pub(super) submit: bool,
    #[arg(long)]
    pub(super) once: bool,
    #[arg(long, default_value_t = 30, value_parser = clap::value_parser!(u64).range(1..))]
    pub(super) interval_seconds: u64,
    #[arg(long, default_value_t = 20_260_729)]
    pub(super) seed_base: u64,
}

#[derive(Clone, Debug, Deserialize)]
pub(super) struct SchedulerJob {
    pub(super) id: String,
    pub(super) project: String,
    pub(super) state: String,
}

#[derive(Clone, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ModelRef {
    pub(super) accepted_parent_checkpoint_sha256: Sha256Digest,
    pub(super) checkpoint_sha256: Sha256Digest,
    pub(super) checkpoint_path: PathBuf,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub(super) struct ActiveIntent {
    pub(super) trainer: TrainerIntentV2,
    pub(super) spec_path: PathBuf,
    pub(super) output_name: String,
    pub(super) warm_start_path: PathBuf,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
#[serde(default)]
pub(super) struct ControllerState {
    pub(super) schema: String,
    pub(super) sequence: u64,
    pub(super) alternation: ShadowAlternationState,
    pub(super) shadow_models: BTreeMap<Task, ModelRef>,
    pub(super) last_shadow_corpus_watermark: BTreeMap<Task, u64>,
    pub(super) last_recipe_e_corpus_watermark: BTreeMap<Task, u64>,
    pub(super) last_recipe_e_descendant_receipts: BTreeMap<Task, u64>,
    pub(super) production_candidates: BTreeMap<Task, ModelRef>,
    pub(super) canary_streaks: TaskCanaryStreaks,
    pub(super) processed_canary_receipts: BTreeSet<String>,
    pub(super) intent: Option<ActiveIntent>,
    pub(super) updated_at_ms: u128,
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
pub(super) struct AcceptedPointer {
    pub(super) digest: Sha256Digest,
    pub(super) path: PathBuf,
}

#[derive(Clone, Debug)]
pub(super) struct CorpusPointer {
    pub(super) watermark: u64,
    pub(super) manifest: PathBuf,
    pub(super) manifest_sha256: String,
    pub(super) packed_manifest: PathBuf,
    pub(super) packed_manifest_sha256: String,
}

pub(crate) fn load_state(path: &Path) -> Result<ControllerState> {
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

pub(crate) fn save_state(path: &Path, state: &ControllerState) -> Result<()> {
    ensure!(
        state.schema == STATE_SCHEMA,
        "task learning state schema drift"
    );
    atomic_json(path, &serde_json::to_value(state)?)
}

pub(crate) fn accepted_pointers(root: &Path) -> Result<BTreeMap<Task, AcceptedPointer>> {
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

pub(crate) fn bootstrap_evaluation_anchor(
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

pub(crate) fn reset_changed_shadow_epochs(
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

#[path = "state/acceptance.rs"]
mod acceptance;
pub(crate) use acceptance::*;

pub(crate) fn promote_task_pointer(
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

#[path = "state/promotion.rs"]
mod promotion;
pub(crate) use promotion::*;
