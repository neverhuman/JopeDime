//! P03 pathway requests: `Direct` and `Thinking` requests with fixed whole-request budgets.
//!
//! This module plans and validates a request. It runs nothing. `Direct` needs checkpoint-bound
//! inference from a published release, and no such runtime exists in this crate yet, so a
//! `Direct` request is refused rather than answered with a silent ridge fallback. `Thinking`
//! is planned as a bounded search that starts from an injected seed, within the level's cap.
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

pub const REQUEST_SCHEMA: &str = "jope.pathway-request/v1";

/// Whole-request evaluation budget caps, counted across every cut, failure, refinement, repeat
/// and compilation of one request.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ThinkingLevel {
    Low,
    Medium,
    High,
    Ultra,
}

impl ThinkingLevel {
    pub const fn cap(self) -> u64 {
        match self {
            Self::Low => 64,
            Self::Medium => 256,
            Self::High => 1024,
            Self::Ultra => 4096,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Mode {
    Direct,
    Thinking(ThinkingLevel),
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PathwayRequest {
    pub schema: String,
    pub release_id: String,
    pub dataset_identity: String,
    pub fit_scope: String,
    pub mode: Mode,
    pub targets: Vec<String>,
    pub seed: u64,
    /// Requested evaluation budget for the whole request. Must not exceed the level's cap.
    pub budget: u64,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PlanError {
    Invalid(String),
    BudgetExceeded {
        requested: u64,
        cap: u64,
    },
    /// No checkpoint-bound inference runtime is available for this release.
    InferenceUnavailable {
        release_id: String,
    },
}

impl std::fmt::Display for PlanError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Invalid(reason) => write!(f, "pathway request invalid: {reason}"),
            Self::BudgetExceeded { requested, cap } => {
                write!(f, "pathway budget {requested} exceeds the level cap {cap}")
            }
            Self::InferenceUnavailable { release_id } => write!(
                f,
                "direct pathway for release {release_id} needs checkpoint-bound inference, which is not available; no fallback is substituted"
            ),
        }
    }
}

impl std::error::Error for PlanError {}

/// A validated `Thinking` plan. The executor starts from the injected seed and stays within
/// `evaluation_budget`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ThinkingPlan {
    pub release_id: String,
    pub dataset_identity: String,
    pub fit_scope: String,
    pub targets: Vec<String>,
    pub seed: u64,
    pub level: ThinkingLevel,
    pub evaluation_budget: u64,
}

pub fn validate(request: &PathwayRequest) -> Result<(), PlanError> {
    if request.schema != REQUEST_SCHEMA {
        return Err(PlanError::Invalid(format!("schema {}", request.schema)));
    }
    for (name, value) in [
        ("release_id", &request.release_id),
        ("dataset_identity", &request.dataset_identity),
        ("fit_scope", &request.fit_scope),
    ] {
        if value.trim().is_empty() {
            return Err(PlanError::Invalid(format!("{name} is empty")));
        }
    }
    if request.targets.is_empty() {
        return Err(PlanError::Invalid("no targets".into()));
    }
    let mut seen = BTreeSet::new();
    for target in &request.targets {
        if target.trim().is_empty() || !seen.insert(target.as_str()) {
            return Err(PlanError::Invalid(format!(
                "target {target:?} is empty or repeated"
            )));
        }
    }
    if request.budget == 0 {
        return Err(PlanError::Invalid("budget is zero".into()));
    }
    Ok(())
}

/// Plans one request. `Direct` is always refused until checkpoint-bound inference exists.
pub fn plan(request: &PathwayRequest) -> Result<ThinkingPlan, PlanError> {
    validate(request)?;
    match request.mode {
        Mode::Direct => Err(PlanError::InferenceUnavailable {
            release_id: request.release_id.clone(),
        }),
        Mode::Thinking(level) => {
            let cap = level.cap();
            if request.budget > cap {
                return Err(PlanError::BudgetExceeded {
                    requested: request.budget,
                    cap,
                });
            }
            Ok(ThinkingPlan {
                release_id: request.release_id.clone(),
                dataset_identity: request.dataset_identity.clone(),
                fit_scope: request.fit_scope.clone(),
                targets: request.targets.clone(),
                seed: request.seed,
                level,
                evaluation_budget: request.budget,
            })
        }
    }
}
