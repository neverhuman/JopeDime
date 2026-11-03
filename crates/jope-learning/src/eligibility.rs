//! Measured train-side OOF admission. A candidate supplies predictions, never eligibility policy
//! or trusted metric values. The policy, targets and trivial baseline are persisted before optimization.
use crate::{
    canon::{canonical_json, canonical_sha256, is_hex64, sha256_hex},
    ledger::{now_unix, Ledger},
    metric::Metric,
    record::{HwmIdentity, HwmRecord},
    state::{PointerKey, State},
};
use anyhow::{bail, ensure, Context, Result};
use jope_datasets::catalog::Task;
use jope_pipeline::evaluation_result::EvaluationResult;
use jope_pipeline::evaluation_score::{score_predictions, Predictions, NUMERICAL_POLICY};
use jope_pipeline::execution::{Artifact, WorkerJob, WorkerOutcome, EXECUTION_SCHEMA};
use jope_pipeline::experiment::{
    EvaluationRequest, EvaluationStage, EvaluationStatus, ExecutionWindow, Experiment, Panel,
    PanelDataset,
};
use jope_pipeline::prepared_data::{parse_development, DevelopmentData};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    io::Write,
    path::Path,
};

mod context;
mod decisions;
mod freeze;
mod inner;
mod measurement;
mod schema;
mod verify;

pub use context::{
    retain_handed_off_context, ContextEvidence, ContextExecution, ContextPlan, ContextRun,
    HandoffEncoder, HandoffEvidence, HandoffRelease, ProducerPlan, HANDOFF_SCHEMA,
    TEACHER_ADMISSION,
};
#[cfg(test)]
use decisions::qualification_path;
pub(crate) use decisions::require_qualified;
pub(super) use decisions::Decisions;
pub use decisions::{freeze_teaching, qualify, revalidate_qualification, revalidate_replay};
#[cfg(test)]
use freeze::freeze_path;
use freeze::{
    baseline, blob, fold_rows, immutable_write, load_freeze, require_blob, rows_sha, task,
};
pub use freeze::{
    effective_label_role, freeze_optimization, freeze_teaching_for, qualify_for,
    require_unburned_test_partition, revalidate_replay_for,
};
pub use inner::{
    derive_inner_partition, freeze_inner_optimization, register_inner_candidate, InnerEvidence,
    InnerPartition,
};
use measurement::measured;
pub(crate) use schema::quarantined;
use schema::{
    diagnostics, requalification_refused, FrozenPolicy, StoredQualification, FREEZE, PREDICTIONS,
    QUALIFICATION, RUST_FITTED, RUST_FITTED_V2,
};
pub use schema::{
    Diagnostics, FoldPredictions, FreezePlan, FreezeReceipt, FrozenFold, GuardPolicy, OofEvidence,
    Qualification, TargetRow, TeachingManifest,
};
#[cfg(test)]
use verify::combine;
use verify::verify_measured_outputs;

#[cfg(test)]
mod tests;
