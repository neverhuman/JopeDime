//! Explicit inner-OOF partition and execution binding. Original development
//! folds remain a different protocol; no old receipt is retroactively upgraded.
use super::*;
use jope_pipeline::{prepared_data::DevelopmentData, search_plan::SelectionPlan};

pub const INNER_SCHEMA: &str = "jope.teaching-inner-partition/v1";
pub const SEED_POLICY: &str =
    "asm.candidate-request-seed/v1:scheduling-only:DSL-and-engine-randomness-pinned";
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `InnerPartition` is a record for the inner partition contract in this module: Explicit inner-OOF partition and execution binding. Original development folds remain a different protocol; no old receipt is retroactively upgraded.
pub struct InnerPartition {
    /// Schema identifier and version for the serialized value.
    pub schema: String,
    /// SHA-256 digest identifying the associated dataset.
    pub dataset_sha256: String,
    /// SHA-256 digest identifying the associated original split.
    pub original_split_sha256: String,
    /// SHA-256 digest identifying the associated development.
    pub development_sha256: String,
    /// Outer fold recorded for this inner partition.
    pub outer_fold: u8,
    /// Number of fitting rows represented here.
    pub fitting_rows: Vec<String>,
    /// Number of hold rows represented here.
    pub hold_rows: Vec<String>,
    /// Collection of fitting groups values carried by this inner partition.
    pub fitting_groups: Vec<String>,
    /// Collection of hold groups values carried by this inner partition.
    pub hold_groups: Vec<String>,
    /// Number of inner validation rows represented here.
    pub inner_validation_rows: Vec<Vec<String>>,
    /// Assignment policy recorded for this inner partition.
    pub assignment_policy: String,
    /// Evaluation seed recorded for this inner partition.
    pub evaluation_seed: u64,
    /// Request seed policy recorded for this inner partition.
    pub request_seed_policy: String,
    /// SHA-256 digest identifying the associated executor program.
    pub executor_program_sha256: String,
    /// SHA-256 digest identifying the associated runtime inventory.
    pub runtime_inventory_sha256: String,
    /// Fit randomness policy recorded for this inner partition.
    pub fit_randomness_policy: String,
    /// Raw input policy recorded for this inner partition.
    pub raw_input_policy: String,
}
impl InnerPartition {
    /// Performs the identity operation and returns an error if it cannot be completed.
    pub fn identity(&self) -> Result<String> {
        canonical_sha256(self)
    }
    /// Performs the fitting rows SHA-256 operation and returns an error if it cannot be completed.
    pub fn fitting_rows_sha256(&self) -> Result<String> {
        canonical_sha256(&self.fitting_rows)
    }
    fn derive(
        data: &DevelopmentData,
        outer_fold: u8,
        seed: u64,
        program: &str,
        runtime: &str,
    ) -> Result<Self> {
        ensure!(
            outer_fold < 5 && is_hex64(program) && is_hex64(runtime),
            "invalid inner partition policy"
        );
        let cut = data.cut(outer_fold)?;
        let rows = cut
            .train
            .rows
            .iter()
            .map(|r| (r.id.clone(), r.group.clone()))
            .collect::<Vec<_>>();
        Ok(Self {
            schema: INNER_SCHEMA.into(),
            dataset_sha256: data.dataset.source.sha256.clone(),
            original_split_sha256: data.dataset.split.sha256.clone(),
            development_sha256: data.dataset.split.development_sha256.clone(),
            outer_fold,
            fitting_rows: cut.train.rows.iter().map(|r| r.id.clone()).collect(),
            hold_rows: cut.hold.rows.iter().map(|r| r.id.clone()).collect(),
            fitting_groups: cut.train.rows.iter().map(|r| r.group.clone()).collect(),
            hold_groups: cut.hold.rows.iter().map(|r| r.group.clone()).collect(),
            inner_validation_rows: jope_pipeline::search_plan::grouped_inner_folds(&rows)?,
            assignment_policy: jope_pipeline::search_plan::INNER_GROUP_POLICY.into(),
            evaluation_seed: seed,
            request_seed_policy: SEED_POLICY.into(),
            executor_program_sha256: program.into(),
            runtime_inventory_sha256: runtime.into(),
            fit_randomness_policy: jope_pipeline::synthesized_recipe::FIT_RANDOMNESS_POLICY.into(),
            raw_input_policy: jope_pipeline::synthesized_recipe::RAW_INPUT_POLICY.into(),
        })
    }
}
fn data(state: &State, panel_sha: &str, id: &str) -> Result<DevelopmentData> {
    let panel: Panel = blob(state, panel_sha)?;
    panel.validate()?;
    let dataset = panel
        .datasets
        .iter()
        .find(|d| d.id == id)
        .context("inner dataset absent from panel")?;
    let bytes = state
        .read_blob(&dataset.split.development_sha256)?
        .context("inner prepared bytes absent")?;
    parse_development(&bytes, dataset)
}
/// Performs the derive inner partition operation and returns an error if it cannot be completed.
pub fn derive_inner_partition(
    state: &State,
    panel_sha: &str,
    id: &str,
    outer_fold: u8,
    seed: u64,
    program: &str,
    runtime: &str,
) -> Result<InnerPartition> {
    InnerPartition::derive(
        &data(state, panel_sha, id)?,
        outer_fold,
        seed,
        program,
        runtime,
    )
}
pub(super) fn authenticated_inner_rows(
    state: &State,
    plan: &FreezePlan,
    partition: &InnerPartition,
) -> Result<(PanelDataset, Vec<TargetRow>)> {
    let source = data(state, &plan.panel_sha256, &plan.dataset_id)?;
    let expected = InnerPartition::derive(
        &source,
        partition.outer_fold,
        plan.evaluation_seed,
        &partition.executor_program_sha256,
        &partition.runtime_inventory_sha256,
    )?;
    ensure!(
        canonical_json(&expected)? == canonical_json(partition)?
            && partition.identity()? == plan.comparison.cut_id
            && source.dataset.source.sha256 == plan.comparison.dataset_hash
            && source.dataset.task.as_str() == plan.comparison.task
            && source.dataset.classes
                == plan
                    .comparison
                    .comparison
                    .as_ref()
                    .context("inner comparison context absent")?
                    .class_order,
        "inner partition differs from actual frozen rows/groups/policy"
    );
    let assignment: BTreeMap<_, _> = partition
        .inner_validation_rows
        .iter()
        .enumerate()
        .flat_map(|(fold, rows)| rows.iter().map(move |r| (r.as_str(), fold as u8)))
        .collect();
    let rows = source
        .rows
        .iter()
        .filter_map(|r| {
            assignment.get(r.id.as_str()).map(|fold| TargetRow {
                id: r.id.clone(),
                group: r.group.clone(),
                fold: *fold,
                target: r.target.clone(),
            })
        })
        .collect::<Vec<_>>();
    ensure!(
        rows.iter().map(|r| &r.id).eq(partition.fitting_rows.iter()),
        "inner target order changed"
    );
    Ok((source.dataset, rows))
}
/// Performs the freeze inner optimization operation and returns an error if it cannot be completed.
pub fn freeze_inner_optimization(
    state: &State,
    plan: &FreezePlan,
    partition: &InnerPartition,
) -> Result<FreezeReceipt> {
    super::freeze_impl(state, plan, Some(partition.clone()))
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
/// `InnerEvidence` is a record for the inner evidence contract in this module: Explicit inner-OOF partition and execution binding. Original development folds remain a different protocol; no old receipt is retroactively upgraded.
pub struct InnerEvidence {
    /// SHA-256 digest identifying the associated selection.
    pub selection_sha256: String,
    /// SHA-256 digest identifying the associated frozen recipe.
    pub frozen_recipe_sha256: String,
    /// SHA-256 digest identifying the associated search receipt.
    pub search_receipt_sha256: String,
    /// SHA-256 digest identifying the associated runtime.
    pub runtime_sha256: String,
    /// SHA-256 digest identifying the associated arm window.
    pub arm_window_sha256: String,
    /// SHA-256 digest identifying the associated memory policy.
    pub memory_policy_sha256: String,
    /// SHA-256 digest identifying the associated candidate freeze.
    pub candidate_freeze_sha256: String,
}
#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct CandidateFreeze {
    schema: String,
    comparison_sha256: String,
    frozen_recipe_sha256: String,
    search_receipt_sha256: String,
    frozen_at_unix_ms: u64,
    selection_source: String,
}
/// Called after measured inner selection and before the controller schedules the
/// candidate's outer prediction job. It records ordering; no outcome is invented.
pub fn register_inner_candidate(
    state: &State,
    freeze: &FreezeReceipt,
    frozen_recipe: &str,
    search_receipt: &str,
) -> Result<String> {
    let policy = load_freeze(state, freeze)?;
    ensure!(
        policy.inner_partition.is_some(),
        "not an inner teaching freeze"
    );
    require_blob(state, frozen_recipe)?;
    require_blob(state, search_receipt)?;
    let path = state
        .root()
        .join("eligibility/candidate-freezes")
        .join(&freeze.comparison_sha256)
        .join(format!("{frozen_recipe}.json"));
    let _lock = state.lock("eligibility")?;
    let ledger = Ledger::open(state.root(), "eligibility");
    ledger.require_intact()?;
    if path.exists() {
        let bytes = fs::read(path)?;
        let value: CandidateFreeze = serde_json::from_slice(&bytes)?;
        ensure!(
            value.comparison_sha256 == freeze.comparison_sha256
                && value.frozen_recipe_sha256 == frozen_recipe
                && value.search_receipt_sha256 == search_receipt,
            "candidate freeze changed"
        );
        return state.put_blob(&bytes);
    }
    let value = CandidateFreeze {
        schema: "jope.inner-candidate-freeze/v1".into(),
        comparison_sha256: freeze.comparison_sha256.clone(),
        frozen_recipe_sha256: frozen_recipe.into(),
        search_receipt_sha256: search_receipt.into(),
        frozen_at_unix_ms: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_millis()
            .try_into()?,
        selection_source: "completed-train-inner-oof-before-candidate-outer-scoring".into(),
    };
    let bytes = canonical_json(&value)?;
    let sha = state.put_blob(&bytes)?;
    immutable_write(&path, &bytes)?;
    ledger.append(&[serde_json::json!({"outcome":"candidate_frozen","comparison_sha256":freeze.comparison_sha256,"candidate_freeze_sha256":sha})],serde_json::json!({"source":"bounded-inner-selection-controller"}))?;
    Ok(sha)
}
pub(super) struct BoundInner {
    pub selection: SelectionPlan,
    pub candidate_sha256: String,
    pub arm: jope_pipeline::execution::ArmBudget,
    pub arm_start: u64,
    pub arm_deadline: u64,
    pub memory: jope_pipeline::execution::FitMemoryAllocation,
    pub frozen_at: u64,
    pub program_sha256: String,
    pub runtime: serde_json::Value,
}
pub(super) fn bind(
    state: &State,
    policy: &FrozenPolicy,
    evidence: &OofEvidence,
    experiment: &Experiment,
    window: &ExecutionWindow,
    live: bool,
) -> Result<Option<BoundInner>> {
    let Some(partition) = &policy.inner_partition else {
        ensure!(
            evidence.inner.is_none(),
            "legacy freeze cannot accept inner evidence"
        );
        return Ok(None);
    };
    let inner = evidence
        .inner
        .as_ref()
        .context("inner execution evidence missing")?;
    ensure!(
        experiment.checkpoints.get(&format!(
            "teaching_freeze/{}",
            evidence.freeze.comparison_sha256
        )) == Some(&evidence.freeze.freeze_sha256)
            && experiment.checkpoints.get(&format!(
                "teaching_vocabulary/{}",
                evidence.freeze.comparison_sha256
            )) == Some(&policy.comparison.vocabulary_sha256),
        "experiment lacks pre-admission inner freeze/vocabulary pins"
    );
    let selection: SelectionPlan = blob(state, &inner.selection_sha256)?;
    selection.validate()?;
    let outer = selection.outer();
    ensure!(
        outer.experiment_sha256 == experiment.identity()?
            && outer.dataset_sha256 == partition.dataset_sha256
            && outer.evaluator_sha256 == policy.comparison.evaluator_sha256
            && outer.task.as_str() == policy.comparison.task
            && outer.split_sha256 == partition.original_split_sha256
            && outer.fold == partition.outer_fold
            && outer.seed == partition.evaluation_seed
            && outer.numerical_policy
                == policy
                    .comparison
                    .comparison
                    .as_ref()
                    .unwrap()
                    .numerical_policy
            && outer.classes == policy.comparison.comparison.as_ref().unwrap().class_order
            && outer.checkpoint_sha256.is_none(),
        "inner outer-template identity changed"
    );
    let expected = SelectionPlan::new(
        outer.clone(),
        partition.fitting_rows.clone(),
        partition.hold_rows.clone(),
        partition.inner_validation_rows.clone(),
        partition.fitting_groups.clone(),
        partition.hold_groups.clone(),
    )?;
    ensure!(
        selection.identity()? == expected.identity()?,
        "inner selection differs from preregistered rows/groups"
    );
    ensure!(
        inner.runtime_sha256 == partition.runtime_inventory_sha256,
        "runtime inventory changed"
    );
    let runtime: serde_json::Value = blob(state, &inner.runtime_sha256)?;
    ensure!(
        runtime.pointer("/program/sha256").and_then(|v| v.as_str())
            == Some(partition.executor_program_sha256.as_str()),
        "inner executor changed"
    );
    for key in ["inputs", "package_files"] {
        let artifacts: Vec<Artifact> = serde_json::from_value(runtime[key].clone())?;
        for artifact in artifacts {
            retain_artifact(state, &artifact, live)?;
        }
    }
    let frozen: serde_json::Value = blob(state, &inner.frozen_recipe_sha256)?;
    ensure!(
        frozen["search_receipt_sha256"] == inner.search_receipt_sha256,
        "selected recipe has different search receipt"
    );
    let receipt: serde_json::Value = blob(state, &inner.search_receipt_sha256)?;
    let candidate = &frozen["candidate"];
    let candidate_sha256 = candidate["id_sha256"]
        .as_str()
        .context("candidate identity absent")?
        .to_owned();
    let recipe: jope_pipeline::synthesized_recipe::SynthesizedRecipe =
        serde_json::from_value(candidate["recipe"].clone())?;
    recipe.validate()?;
    ensure!(
        candidate["program_sha256"] == recipe.executable_sha256
            && frozen["selection_feedback"]["candidate_sha256"] == candidate_sha256
            && frozen["selection_feedback"]["status"] == "completed",
        "frozen candidate is not a complete typed selection"
    );
    let observations = receipt["observations"]
        .as_array()
        .context("selection observations absent")?;
    ensure!(
        observations
            .iter()
            .any(|o| o["candidate"] == *candidate && o["feedback"] == frozen["selection_feedback"]),
        "frozen candidate absent from actual selection receipt"
    );
    let marker: CandidateFreeze = blob(state, &inner.candidate_freeze_sha256)?;
    ensure!(
        marker.schema == "jope.inner-candidate-freeze/v1"
            && marker.comparison_sha256 == evidence.freeze.comparison_sha256
            && marker.frozen_recipe_sha256 == inner.frozen_recipe_sha256
            && marker.search_receipt_sha256 == inner.search_receipt_sha256
            && Ledger::open(state.root(), "eligibility")
                .events()?
                .iter()
                .any(|e| e["outcome"] == "candidate_frozen"
                    && e["candidate_freeze_sha256"] == inner.candidate_freeze_sha256),
        "candidate lacks pre-outer-scoring ledger record"
    );
    let arm: jope_pipeline::execution::ArmBudget = serde_json::from_value(receipt["arm"].clone())?;
    let arm_window: serde_json::Value = blob(state, &inner.arm_window_sha256)?;
    let arm_start = arm_window["started_unix_ms"]
        .as_u64()
        .context("arm start absent")?;
    let arm_deadline = arm_window["deadline_unix_ms"]
        .as_u64()
        .context("arm deadline absent")?;
    ensure!(
        (1..=120).contains(&arm.seconds)
            && is_hex64(&arm.id_sha256)
            && arm_window["id_sha256"] == arm.id_sha256
            && arm_window["seconds"] == arm.seconds
            && arm_start.checked_add(arm.seconds * 1000) == Some(arm_deadline)
            && arm_start >= window.started_unix_ms,
        "arm window changed"
    );
    let cutoff =
        arm_deadline.min(window.deadline_unix_ms - experiment.resources.reporting_seconds * 1000);
    ensure!(
        receipt["deadline_unix_ms"] == cutoff,
        "selection receipt changed original cutoff"
    );
    let memory: jope_pipeline::execution::FitMemoryAllocation =
        blob(state, &inner.memory_policy_sha256)?;
    ensure!(
        memory == jope_pipeline::execution::fit_memory_allocation(experiment)?,
        "inner memory policy changed"
    );
    Ok(Some(BoundInner {
        selection,
        candidate_sha256,
        arm,
        arm_start,
        arm_deadline: cutoff,
        memory,
        frozen_at: marker.frozen_at_unix_ms,
        program_sha256: recipe.executable_sha256,
        runtime,
    }))
}
pub(super) fn validate_execution(
    binding: &BoundInner,
    request: &EvaluationRequest,
    job: &WorkerJob,
    worker: &WorkerOutcome,
    started_ms: u64,
    deadline_ms: u64,
    program: &str,
) -> Result<()> {
    ensure!(
        program == binding.program_sha256,
        "selected candidate program changed"
    );
    let runtime_program: Artifact = serde_json::from_value(binding.runtime["program"].clone())?;
    let prefix: Vec<String> = serde_json::from_value(binding.runtime["argument_prefix"].clone())?;
    let environment: BTreeMap<String, String> =
        serde_json::from_value(binding.runtime["environment"].clone())?;
    ensure!(
        runtime_program.path == job.command.program.path
            && runtime_program.sha256 == job.command.program.sha256
            && job.command.args.starts_with(&prefix)
            && environment == job.command.environment,
        "fit runtime differs from full preregistered identity"
    );
    let inputs: Vec<Artifact> = serde_json::from_value(binding.runtime["inputs"].clone())?;
    ensure!(
        inputs.iter().all(|a| job
            .command
            .inputs
            .iter()
            .any(|b| a.path == b.path && a.sha256 == b.sha256)),
        "runtime input closure changed"
    );
    let expected = binding
        .selection
        .selection_requests(program, &binding.candidate_sha256)?;
    ensure!(
        expected
            .get(request.fold as usize)
            .context("inner fold absent")?
            .identity()?
            == request.identity()?
            && job.arm.as_ref().is_some_and(
                |a| a.id_sha256 == binding.arm.id_sha256 && a.seconds == binding.arm.seconds
            ),
        "inner request or arm changed"
    );
    let gate = worker
        .consumption
        .gate_released_unix_ms
        .context("inner gate unknown")?;
    let exit = worker
        .consumption
        .observed_exit_unix_ms
        .context("inner exit unknown")?;
    ensure!(
        started_ms >= binding.arm_start
            && gate >= started_ms
            && exit >= gate
            && exit <= deadline_ms
            && deadline_ms <= binding.arm_deadline
            && exit <= binding.frozen_at
            && worker.consumption.fit_memory_limit_bytes == Some(binding.memory.per_fit_bytes),
        "inner fit escaped original admission or completed after candidate freeze"
    );
    Ok(())
}
