//! The four continuation modes (W6-T06).
//!
//! Every training attempt is exactly one of:
//! - **cold initialization**: no parent; a new lineage.
//! - **weight warm start**: parent weights only. The optimizer is reset and
//!   the sampler starts a new window at zero. The objective may change;
//!   the change is recorded.
//! - **new-corpus transition**: parent weights and carried optimizer
//!   moments, with a new sampling window, policy or corpus. The objective
//!   must be the same. An execution-partition change must be declared.
//! - **exact resume**: the complete same-lineage state. Nothing may change.
//!
//! Vocabulary and architecture never change implicitly in any mode. A
//! parent written before the objective identity existed cannot prove its
//! objective: it is refused for exact resume and optimizer carry, and
//! continues only as a weight warm start, recorded as `legacy_ids_only` when
//! its two legacy objective identifiers match and never as `identical`
//! (owner decision 2026-09-26, `docs/internal/core-ip/work-orders/OWNER-DECISIONS.md` D12).
use super::objective::{identity_digest, RUNTIME_OBJECTIVE_ID};
use anyhow::{ensure, Context, Result};
use serde_json::{json, Value};

/// The mode vocabulary and classification are shared with request types
/// outside the neural feature (`task_contract`).
pub use crate::task_contract::{classify_continuation as classify, ContinuationMode as Mode};

pub const SCHEMA: &str = "jope.continuation.v1";

/// Additive bridge for a verified completed edge, or an exact attempt resume at
/// its tip. The original mode/objective/topology guards execute before this bridge.
pub fn check_versioned_parent(
    mode: Mode,
    state: &Value,
    config: &Value,
    current: &Current,
    policy: Option<&Value>,
    parent_checkpoint_sha256: &str,
    lineage: &crate::native_loop::lineage::VerifiedLineage,
) -> Result<Value> {
    let mut report = check_parent(mode, state, config, current, policy)?;
    let edge = lineage.tip();
    let partition = crate::task_contract::training_partition(
        current.world,
        current.microbatch,
        current.global_batch,
    )?;
    ensure!(
        current.objective_sha256 == edge.intent.protocol.objective_sha256
            && current.vocabulary_sha256 == edge.intent.protocol.vocabulary_sha256
            && current.corpus_sha256 == edge.intent.corpus_sha256
            && identity_digest(current.config) == edge.intent.protocol.architecture_sha256
            && identity_digest(&partition) == edge.intent.execution_partition_sha256
            && crate::manifest::hash(current.sampler_id.as_bytes()) == edge.intent.sampler_sha256,
        "versioned continuation differs from verified intent"
    );
    if mode == Mode::ExactResume {
        ensure!(
            parent_checkpoint_sha256 == edge.checkpoint_sha256
                && state["step"].as_u64() == Some(lineage.local_step())
                && state["model_sha256"] == edge.weights_sha256
                && state["optimizer_sha256"] == edge.optimizer_sha256
                && state["optimizer_recipe_sha256"] == edge.epoch.recipe_sha256
                && state["optimizer_epoch_sha256"] == edge.intent.optimizer_epoch_sha256,
            "exact resume differs from verified checkpoint/optimizer epoch"
        );
    } else {
        ensure!(
            mode == edge.mode && parent_checkpoint_sha256 == edge.parent_checkpoint_sha256,
            "continuation mode/parent differs from verified ancestry edge"
        );
        let prior = lineage.edges().iter().rev().nth(1);
        let (step, weights) = prior.map_or(
            (
                lineage.lineage().anchor.optimizer_step,
                &lineage.lineage().anchor.weights_sha256,
            ),
            |p| (p.intent.target_step, &p.weights_sha256),
        );
        ensure!(
            state["step"].as_u64() == Some(step) && state["model_sha256"] == *weights,
            "continuation parent weights/step differ from verified ancestry"
        );
        if mode == Mode::NewCorpusTransition {
            ensure!(
                prior.is_some_and(|p| state["optimizer_sha256"] == p.optimizer_sha256
                    && state["optimizer_recipe_sha256"] == p.epoch.recipe_sha256
                    && state["optimizer_epoch_sha256"] == p.intent.optimizer_epoch_sha256),
                "optimizer carry requires actual verified parent epoch/state"
            );
        }
    }
    report["policy_lineage"] = json!({"schema": crate::native_loop::lineage::LINEAGE_SCHEMA,
        "anchor_checkpoint_sha256": lineage.lineage().anchor.checkpoint_sha256,
        "optimizer_epoch_sha256": edge.intent.optimizer_epoch_sha256,
        "local_step": lineage.local_step(), "committed_additional_updates": lineage.committed_updates(),
        "cumulative_committed_updates": lineage.cumulative_updates()});
    Ok(report)
}

/// Validate a request's optional lineage-mode declaration.
///
/// The lineage mode is how the run's lineage started: cold, warm or
/// transition. For a fresh attempt it is the classified request. For a
/// resume it is the mode recorded in the lineage's `initialization.json`,
/// supplied as `lineage`.
///
/// - An explicit declaration must equal the lineage mode. `ExactResume` is
///   never a lineage declaration, and a legacy lineage without a recorded mode
///   cannot be declared into certainty.
/// - A request without a declaration keeps its implicit meaning and is
///   recorded as `implicit`.
pub fn declared_lineage(
    declared: Option<Mode>,
    resume: bool,
    parent: bool,
    policy: Option<&Value>,
    lineage: Option<&Value>,
) -> Result<Value> {
    let attempt = classify(resume, parent, policy)?;
    let lineage_mode = if resume {
        let record = lineage.context("a resume must bring its lineage record")?;
        record
            .get("continuation_mode")
            .and_then(Value::as_str)
            .map(Mode::parse)
            .transpose()?
    } else {
        Some(attempt)
    };
    if let Some(declared) = declared {
        ensure!(
            declared != Mode::ExactResume,
            "exact resume is an attempt property; declare the lineage mode"
        );
        let recorded =
            lineage_mode.context("a declared mode cannot be verified against a legacy lineage")?;
        ensure!(
            declared == recorded,
            "declared continuation mode {} differs from the request's {}",
            declared.as_str(),
            recorded.as_str()
        );
    }
    Ok(json!({
        "attempt_mode": attempt.as_str(),
        "lineage_mode": lineage_mode.map_or("unknown_legacy", Mode::as_str),
        "declared": declared.map(Mode::as_str),
        "declaration": if declared.is_some() { "explicit" } else { "implicit" },
    }))
}

/// The attempt being started.
pub struct Current<'a> {
    pub corpus_sha256: &'a str,
    pub vocabulary_sha256: &'a str,
    pub world: usize,
    pub microbatch: usize,
    pub global_batch: usize,
    pub seed: u64,
    pub sampler_id: &'a str,
    /// Serialized architecture configuration.
    pub config: &'a Value,
    pub objective_sha256: &'a str,
}

/// Validate the parent checkpoint `state` (with its serialized `config`)
/// against the requested mode. Returns the continuity record: what
/// continued, what changed and was declared or permitted, and what is
/// unknown. `policy` may declare `{"declared_changes": {"execution_partition": ...}}`
/// for an optimizer-carry transition; nothing else can be declared.
pub fn check_parent(
    mode: Mode,
    state: &Value,
    config: &Value,
    current: &Current,
    policy: Option<&Value>,
) -> Result<Value> {
    let partition = crate::task_contract::training_partition(
        current.world,
        current.microbatch,
        current.global_batch,
    )?;
    check_parent_partition(mode, state, config, current, &partition, policy)
}

/// [`check_parent`] for an explicit execution-partition record, v1 or v2
/// (`training::partition::Plan::record`, W7-T06). A different rank count or
/// microbatch is a changed partition: exact resume refuses it, an
/// optimizer-carry transition must declare it, and a weight warm start
/// carries nothing.
pub fn check_parent_partition(
    mode: Mode,
    state: &Value,
    config: &Value,
    current: &Current,
    partition: &Value,
    policy: Option<&Value>,
) -> Result<Value> {
    ensure!(
        partition["world"].as_u64() == Some(current.world as u64)
            && partition["global_batch"].as_u64() == Some(current.global_batch as u64),
        "execution partition differs from the requested topology"
    );
    ensure!(
        mode != Mode::ColdInitialization,
        "cold initialization has no parent"
    );
    ensure!(
        state["vocabulary_sha256"] == current.vocabulary_sha256,
        "continuation across a vocabulary change is refused; no token remapping is implicit"
    );
    ensure!(
        config == current.config,
        "continuation across an architecture change is refused"
    );
    let parent_objective = match state.get("objective_sha256").filter(|v| !v.is_null()) {
        Some(digest) => {
            let digest = digest
                .as_str()
                .ok_or_else(|| anyhow::anyhow!("parent objective digest malformed"))?;
            ensure!(
                identity_digest(&state["objective"]) == digest,
                "parent objective identity does not match its digest"
            );
            Some(digest)
        }
        None => None,
    };
    let legacy_ids = state["ranking_backward"] == crate::neural::RANKING_BACKWARD_ID
        && state["runtime_objective"] == RUNTIME_OBJECTIVE_ID;
    let objective = match parent_objective {
        Some(digest) if digest == current.objective_sha256 => "identical",
        Some(_) => "changed",
        None if legacy_ids => "legacy_ids_only",
        None => "changed",
    };
    // A legacy parent's objective is unverified: nothing but its weights may
    // continue (OWNER-DECISIONS.md D12). Checked before any other mode rule.
    ensure!(
        objective != "legacy_ids_only" || mode == Mode::WeightWarmStart,
        "a parent without an objective identity continues only as a weight warm start \
         (optimizer reset); {} is refused",
        mode.as_str()
    );
    let parent_partition = state
        .get("execution_partition")
        .filter(|v| !v.is_null())
        .cloned();
    let partition_continuity = match &parent_partition {
        Some(p) if p == partition => "identical",
        Some(_) => "changed",
        None => "unknown",
    };
    let declared = policy
        .and_then(|p| p.get("declared_changes"))
        .cloned()
        .unwrap_or_else(|| json!({}));
    let declared_fields = declared
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("declared changes must be an object"))?;
    ensure!(
        declared_fields.keys().all(|k| k == "execution_partition"),
        "only an execution-partition change can be declared"
    );
    let declared_partition = declared.get("execution_partition");
    ensure!(
        declared_partition.is_none_or(|p| p == partition),
        "declared execution partition differs from the requested topology"
    );
    match mode {
        Mode::ColdInitialization => unreachable!(),
        Mode::ExactResume => {
            ensure!(
                declared_fields.is_empty(),
                "exact resume cannot declare changes"
            );
            ensure!(
                objective != "changed",
                "exact resume requires the same objective; use an explicit weight warm start"
            );
            ensure!(
                partition_continuity == "identical",
                "exact resume requires the original microbatch, accumulation and RNG/objective partition"
            );
            ensure!(
                state["corpus_sha256"] == current.corpus_sha256
                    && state["world"].as_u64() == Some(current.world as u64)
                    && state["global_batch"].as_u64() == Some(current.global_batch as u64)
                    && state["seed"].as_u64() == Some(current.seed)
                    && state["sampler_id"] == current.sampler_id,
                "exact resume requires the same corpus, topology, seed and sampler"
            );
        }
        Mode::NewCorpusTransition => {
            ensure!(
                objective != "changed",
                "optimizer carry requires the same objective; an objective change needs a weight warm start with optimizer reset"
            );
            ensure!(
                partition_continuity != "changed" || declared_partition.is_some(),
                "undeclared execution-partition change under optimizer carry"
            );
        }
        Mode::WeightWarmStart => {
            ensure!(
                declared_fields.is_empty(),
                "a weight warm start carries nothing; it declares no changes"
            );
        }
    }
    Ok(json!({
        "schema": SCHEMA,
        "mode": mode.as_str(),
        "lineage_mode": state["initialization"]
            .get("continuation_mode")
            .cloned()
            .unwrap_or_else(|| json!("unknown_legacy")),
        "exact_resume_claim": mode == Mode::ExactResume,
        "optimizer": if mode == Mode::WeightWarmStart { "reset" } else { "carried" },
        "sampler": if mode == Mode::ExactResume { "continued" } else { "new_window_from_zero" },
        "objective": {
            "parent_sha256": parent_objective,
            "current_sha256": current.objective_sha256,
            "continuity": objective,
        },
        "execution_partition": {
            "parent": parent_partition,
            "current": partition,
            "continuity": partition_continuity,
            "declared": declared_partition,
        },
        "corpus": {
            "parent": state["corpus_sha256"],
            "current": current.corpus_sha256,
        },
        "parent_step": state["step"],
    }))
}
