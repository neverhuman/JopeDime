//! Typed payloads inside the frozen WRK-02 framing. No matrix bytes cross the pipe.

use jope_api::ids::RunId;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use std::path::PathBuf;

/// Complete controller fence, echoed unchanged by the worker.
#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Attempt {
    pub run: RunId,
    pub round: u64,
    pub slot: u32,
    pub attempt: u64,
}

impl From<&crate::run::AttemptKey> for Attempt {
    fn from(key: &crate::run::AttemptKey) -> Self {
        Self {
            run: key.run.clone(),
            round: key.round,
            slot: key.slot,
            attempt: key.attempt,
        }
    }
}

#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Limits {
    pub wall_ms: u64,
    pub rss_bytes: u64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Request {
    pub attempt: Attempt,
    pub matrix_path: PathBuf,
    pub fold_spec: Value,
    pub dsl: String,
    pub objective: Value,
    pub seed: u64,
    pub limits: Limits,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WorkerFailure {
    Timeout,
    MemoryLimit,
    Panicked,
    NonFinite,
    ProcessExited,
    Protocol,
    InvalidRequest,
    Unavailable,
    StaleAttempt,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Response {
    pub attempt: Attempt,
    pub scores: Vec<f64>,
    /// Per-fold objective values; omitted by older workers and non-native engines.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub fold_scores: Vec<f64>,
    pub predictions_digest: String,
    pub wall_ms: u64,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure: Option<WorkerFailure>,
}

impl Response {
    pub fn failed(attempt: Attempt, failure: WorkerFailure, wall_ms: u64) -> Self {
        Self {
            attempt,
            scores: Vec::new(),
            fold_scores: Vec::new(),
            predictions_digest: String::new(),
            wall_ms,
            failure: Some(failure),
        }
    }

    pub(crate) fn valid(&self) -> bool {
        if self.failure.is_some() {
            self.scores.is_empty() && self.predictions_digest.is_empty()
        } else {
            !self.scores.is_empty()
                && self.scores.iter().all(|v| v.is_finite())
                && self.fold_scores.iter().all(|v| v.is_finite() && *v >= 0.0)
                && self.predictions_digest.len() == 64
                && self
                    .predictions_digest
                    .bytes()
                    .all(|b| b.is_ascii_hexdigit())
        }
    }
}

/// Evaluator output; the worker checks finiteness and the digest before responding.
pub struct Scores {
    pub values: Vec<f64>,
    pub fold_scores: Vec<f64>,
    pub predictions_digest: String,
}
