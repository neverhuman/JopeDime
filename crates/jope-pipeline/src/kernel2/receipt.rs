//! Opt-in profile-search receipts. Legacy receipt types and schemas stay separate.

use anyhow::{ensure, Context, Result};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::{collections::BTreeMap, path::Path};

use super::{
    records::Measurement,
    search::{PolicyRecord, Refinement},
    EngineRecord, Kernel2Evaluator,
};
use crate::{receipt as receipt_v3, search};

pub const SCHEMA_ID: &str = "jope.pipeline-receipt/v4";
pub const SCHEMA_JSON: &str =
    include_str!("../../../../schemas/jope.pipeline-receipt.v4.schema.json");
pub const ENGINE: &str = "feat-math profile-pipeline GP (in-process)";
pub const SELECTION: &str = "kernel2_fault_aware_validation_score_v1";

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Search {
    pub engine: String,
    pub profile_id: String,
    pub profile_sha256: String,
    pub fault_policy: PolicyRecord,
    pub rng_seed: u64,
    pub evaluations: usize,
    pub seconds: u64,
    pub refinement: Refinement,
    pub cut: usize,
    pub train_rows: usize,
    pub hold_rows: usize,
    pub generations: usize,
    pub completed_evaluations: usize,
    pub candidate_measurements: usize,
    pub objective_trials: usize,
    pub rejected_trials: usize,
    pub successful_measurements: usize,
    pub stopped_by: String,
    pub active_seconds: f64,
    pub deadline_overrun: bool,
    pub arm_counts: BTreeMap<String, usize>,
    pub seeding: receipt_v3::Seeding,
    pub selection: String,
    pub best: Measurement,
    pub observations_sha256: String,
    pub refinements_sha256: String,
    /// Opt-in engine methods. Absent for the default selection, so its receipts keep their bytes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub engine_selection: Option<EngineRecord>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct Receipt {
    pub schema: String,
    pub ok: bool,
    pub tool_version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub git_sha: Option<String>,
    pub created_at: String,
    pub policy: String,
    pub dataset: receipt_v3::Dataset,
    pub prepared: receipt_v3::Prepared,
    pub stages: Vec<receipt_v3::Stage>,
    pub search: Search,
    pub artifacts: Vec<receipt_v3::Artifact>,
    pub emit_exclusions: Vec<receipt_v3::Exclusion>,
    pub search_sha256: String,
    pub emit_manifest_sha256: String,
}

pub fn schema() -> serde_json::Value {
    serde_json::from_str(SCHEMA_JSON).expect("committed v4 schema parses")
}

pub fn derived_schema() -> serde_json::Value {
    let mut schema = schemars::schema_for!(Receipt).to_value();
    schema["title"] = SCHEMA_ID.into();
    schema["$id"] =
        "https://github.com/neverhuman/JopeDime/schemas/jope.pipeline-receipt.v4.schema.json"
            .into();
    schema["properties"]["schema"]["const"] = SCHEMA_ID.into();
    schema["properties"]["policy"]["enum"] = serde_json::json!(["auto", "skip"]);
    let shared_schema = receipt_v3::derived_schema();
    for name in [
        "Dataset",
        "DatasetFile",
        "Prepared",
        "Stage",
        "Artifact",
        "Exclusion",
        "Seeding",
        "Seed",
    ] {
        if let Some(shared) = shared_schema["$defs"].get(name) {
            schema["$defs"][name] = shared.clone();
        } else {
            schema["$defs"][name]["additionalProperties"] = false.into();
        }
    }
    schema
}

impl Search {
    pub fn evaluator(&self) -> Result<Kernel2Evaluator> {
        let profile = feat_math::kernel::profile::shipped_kernel2_profiles()
            .into_iter()
            .find(|p| p.id == self.profile_id)
            .context("unknown shipped profile")?;
        ensure!(
            profile.sha256() == self.profile_sha256,
            "profile digest does not match shipped profile"
        );
        Kernel2Evaluator::new(profile)?.with_fault_policy(self.fault_policy.policy()?)
    }
}

pub fn validate_value(value: &serde_json::Value) -> Result<Receipt> {
    let errors = jope_datasets::schema::validate(&schema(), value);
    ensure!(
        errors.is_empty(),
        "invalid v4 receipt: {}",
        errors.join("; ")
    );
    let receipt: Receipt = serde_json::from_value(value.clone())?;
    let s = &receipt.search;
    search::Budget {
        evaluations: s.evaluations,
        seconds: s.seconds,
    }
    .check()?;
    ensure!(
        receipt.schema == SCHEMA_ID
            && s.engine == ENGINE
            && s.selection == SELECTION
            && receipt.dataset.task == "regression"
            && receipt.prepared.cuts == [0]
            && s.cut == 0
            && s.train_rows >= 2
            && s.hold_rows >= 2
            && s.train_rows.checked_add(s.hold_rows) == Some(receipt.prepared.rows as usize)
            && receipt.prepared.rows == receipt.dataset.training_rows,
        "invalid v4 run identity or cut contract"
    );
    ensure!(
        s.completed_evaluations > 0
            && s.completed_evaluations <= s.evaluations
            && s.candidate_measurements.checked_add(s.objective_trials)
                == Some(s.completed_evaluations)
            && s.rejected_trials <= s.objective_trials
            && s.successful_measurements > 0
            && s.successful_measurements <= s.candidate_measurements
            && s.refinement.top_k <= search::ELITES
            && s.refinement.trials_per_program <= s.evaluations
            && s.generations > 0
            && s.active_seconds.is_finite()
            && s.active_seconds >= 0.0
            && s.deadline_overrun == (s.active_seconds > s.seconds as f64),
        "invalid shared budget accounting"
    );
    ensure!(
        [
            search::STOPPED_BY_EVALUATIONS,
            search::STOPPED_BY_SECONDS,
            search::STOPPED_BY_EXHAUSTED
        ]
        .contains(&s.stopped_by.as_str()),
        "unknown stop reason"
    );
    if s.stopped_by == search::STOPPED_BY_EVALUATIONS {
        ensure!(
            s.completed_evaluations == s.evaluations,
            "evaluation stop before the bound"
        );
    } else if s.stopped_by == search::STOPPED_BY_SECONDS {
        ensure!(
            s.active_seconds >= s.seconds as f64,
            "seconds stop before the bound"
        );
    }
    // laneD:B17-lm-refinement begin
    if let Some(engine) = &s.engine_selection {
        engine.selection()?.validate(s.refinement)?;
    }
    // laneD:B17-lm-refinement end
    let evaluator = s.evaluator()?;
    let best = s.best.derive(evaluator.fault_policy())?;
    ensure!(
        best.observation.measured()
            && best.observation.cut == s.cut
            && best.observation.generation < s.generations,
        "invalid selected measurement"
    );
    // laneD:B18-pareto-archive begin
    let archive_sha256 = s
        .engine_selection
        .as_ref()
        .and_then(|e| e.archive.as_ref())
        .map(|a| {
            a.sha256
                .as_ref()
                .context("a published archive needs its digest")
        })
        .transpose()?;
    // laneD:B18-pareto-archive end
    // laneD:B20-rewrite-simplify begin
    let moves_sha256 = s
        .engine_selection
        .as_ref()
        .and_then(|e| e.moves.as_ref())
        .map(|m| {
            m.sha256
                .as_ref()
                .context("published move records need their digest")
        })
        .transpose()?;
    // laneD:B20-rewrite-simplify end
    for digest in [
        &receipt.search_sha256,
        &s.observations_sha256,
        &s.refinements_sha256,
        &receipt.emit_manifest_sha256,
    ]
    .into_iter()
    .chain(archive_sha256)
    .chain(moves_sha256)
    {
        ensure!(
            digest.len() == 64
                && digest
                    .bytes()
                    .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)),
            "invalid evidence digest"
        );
    }
    ensure!(
        receipt.policy != "require" && receipt.ok == !receipt.artifacts.is_empty(),
        "invalid run outcome"
    );
    for name in [
        "dataset", "prepare", "propose", "search", "select", "emit", "receipt",
    ] {
        let stages: Vec<_> = receipt.stages.iter().filter(|s| s.name == name).collect();
        ensure!(
            stages.len() == 1
                && receipt_v3::STATUSES.contains(&stages[0].status.as_str())
                && stages[0].seconds.is_finite()
                && stages[0].seconds >= 0.0
                && (stages[0].status == "passed"
                    || stages[0].reason.as_ref().is_some_and(|r| !r.is_empty())),
            "invalid or missing stage {name}"
        );
        let expected = if name == "propose" {
            "skipped"
        } else if name == "emit" && !receipt.ok {
            "failed"
        } else {
            "passed"
        };
        ensure!(
            stages[0].status == expected,
            "stage {name} does not match its outcome"
        );
    }
    ensure!(receipt.stages.len() == 7, "unknown stage");
    Ok(receipt)
}

pub fn read(path: &Path) -> Result<Receipt> {
    let bytes = std::fs::read(path).with_context(|| format!("read {}", path.display()))?;
    validate_value(&serde_json::from_slice(&bytes)?)
}
