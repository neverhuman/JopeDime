//! Finalization of the numeric M1 champion. Prepared partitions are read only after
//! the durable once-only claim and the row-audit finalization boundary.
//!
//! Keep one `Finalizer` per store and share `journal()` with its subscribers. HTTP,
//! CLI and constrained-split preparation remain adapters; this API never searches.
mod model;
pub(super) use model::{load, predict, publish, put_frozen_model, refit, score};

use super::{
    audit::{AuditRowId, RowAudit, RowAuditPhase},
    r#loop::{Champion, Executor, Plan},
};
use crate::{
    dataset::jmat::{JmatCell, Matrix},
    events::{EventLog, RunUpdate},
    objective::regression::RegressionObjective,
    store::{BlobRef, Record, RecordKey, Store, StoreError},
    worker::{Attempt, Request, WorkerFailure},
};
use anyhow::{bail, ensure, Context, Result};
use jope_api::{
    ids::{ArtifactId, RunId},
    CompletionReason, EventBody, FinalScores, Refit, RunFinishedBody, RunPhase, RunSnapshot,
    RunState,
};
use jope_runtime::{
    math_pipeline::{
        InputContract, NumericColumn, PipelineState, TargetScale, INPUT_PART, PIPELINE_PART,
    },
    model_object::{Manifest, ModelObject, Part, Producer, MATH_PIPELINE, SCHEMA},
    table::CsvTable,
};
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::{collections::BTreeMap, fs, io::Read, path::PathBuf, sync::Arc};

const HANDOFF_MAX: u64 = 16 * 1024 * 1024;

/// Every file contains exactly these dataset-local source rows in this order.
/// The split preparer supplies the provenance roster; no payload is read here.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Partition {
    pub path: PathBuf,
    pub sha256: String,
    pub rows: Vec<AuditRowId>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FinalizationRequest {
    pub run_id: RunId,
    pub fit: Partition,
    pub holdout: Partition,
    pub test: Option<Partition>,
    pub refit: Refit,
    /// Runtime model-object root, not a client-selected output file.
    pub model_root: PathBuf,
    pub scratch_root: PathBuf,
    pub max_input_bytes: u64,
    pub max_model_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FinalizationReceipt {
    pub run_id: RunId,
    pub algorithm_id: jope_api::ids::AlgorithmId,
    pub model_artifact: ArtifactId,
    pub model_digest: String,
    pub model_path: PathBuf,
    pub refit: Refit,
    pub scores: FinalScores,
    pub completion_reason: CompletionReason,
    pub budget_bound_by: Option<String>,
    pub budget_used: jope_api::BudgetUsed,
    pub nondeterminism: jope_api::Nondeterminism,
    /// HOLDOUT describes this FIT-only model, even when the export is refitted.
    pub holdout_model_digest: String,
    pub holdout_scope: String,
    pub test_scope: Option<String>,
    pub fit_rows: usize,
    pub holdout_rows: usize,
    pub test_rows: usize,
    pub audit: Value,
}

pub struct Finalizer {
    store: Arc<Store>,
    journal: Arc<EventLog>,
}
mod execute;

// Qualification may attach a measured receipt after completion. Repeating an
// identical native model in another run preserves that attachment; every frozen
// field and blob must still match exactly.
#[cfg(test)]
mod tests;
