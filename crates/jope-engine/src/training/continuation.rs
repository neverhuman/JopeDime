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
//! parent written before the objective identity existed is accepted only
//! when its two legacy objective identifiers match, and it is recorded as
//! `legacy_ids_only`, never as `identical`.
use super::objective::{identity_digest, RUNTIME_OBJECTIVE_ID};
use anyhow::{bail, ensure, Result};
use serde_json::{json, Value};

pub const SCHEMA: &str = "jope.continuation.v1";

#[derive(Clone, Copy, Debug, PartialEq, Eq, serde::Serialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    ColdInitialization,
    WeightWarmStart,
    NewCorpusTransition,
    ExactResume,
}

impl Mode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ColdInitialization => "cold_initialization",
            Self::WeightWarmStart => "weight_warm_start",
            Self::NewCorpusTransition => "new_corpus_transition",
            Self::ExactResume => "exact_resume",
        }
    }
}

/// The mode a request asks for. Ambiguous or incomplete requests are
/// refused rather than mapped to a nearby mode.
pub fn classify(resume: bool, parent: bool, policy: Option<&Value>) -> Result<Mode> {
    ensure!(
        !(resume && parent),
        "resume and initialize are mutually exclusive"
    );
    if resume {
        return Ok(Mode::ExactResume);
    }
    match (parent, policy) {
        (false, None) => Ok(Mode::ColdInitialization),
        (false, Some(_)) => {
            bail!("a training policy requires an explicit parent checkpoint or exact resume")
        }
        (true, None) => Ok(Mode::WeightWarmStart),
        (true, Some(policy)) => match policy["optimizer"].as_str() {
            Some("reset") => Ok(Mode::WeightWarmStart),
            Some("carry") => Ok(Mode::NewCorpusTransition),
            _ => bail!("a training policy must declare optimizer reset or carry"),
        },
    }
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
    let partition = crate::task_contract::training_partition(
        current.world,
        current.microbatch,
        current.global_batch,
    )?;
    let parent_partition = state
        .get("execution_partition")
        .filter(|v| !v.is_null())
        .cloned();
    let partition_continuity = match &parent_partition {
        Some(p) if *p == partition => "identical",
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
        declared_partition.is_none_or(|p| *p == partition),
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
