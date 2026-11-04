//! Task-aware admission preflight for evaluator manifests.
//!
//! The generated operator registry is the operator/task authority. This
//! module deliberately does not carry a second handwritten operator list.

mod evaluate;
mod outcome;
mod registry;
#[cfg(test)]
mod tests;

pub use evaluate::{preflight_task_manifest, TaskPreflightRequest};
pub use outcome::{
    TaskPreflightDisposition, TaskPreflightOutcomeV1, TaskPreflightReason, TaskPreflightWorkV1,
    TASK_PREFLIGHT_OUTCOME_SCHEMA,
};
pub use registry::{
    GeneratedOperatorRegistry, TASK_HEAD_POLICY_JSON, TASK_HEAD_POLICY_SCHEMA,
    TASK_HEAD_POLICY_SHA256,
};
