//! Rust port of the scoring and deployment policy used by Python's
//! `jope.self_play_evaluation` mixin.
//!
//! The Python orchestration remains the active adapter while
//! `self_play.py` imports `SelfPlayEvaluationMixin` (PORT-KEEP). This module
//! keeps the data transformations and decision formulas available to Rust
//! callers without adding Python, ML, or filesystem dependencies to the policy.

mod evaluation;
mod gate;
mod helpers;

pub use evaluation::{
    evaluate_and_append, DatasetCandidates, DatasetContext, EvaluationInput, EvaluationRequest,
    EvaluationSummary, ScoredEvent,
};
pub use gate::{
    benchmark_summary, deployment_elo, gate_generation, last_shipped_deployed_scores,
    resolve_ship_decision, GateState, GateSummary,
};
pub use helpers::robust_score;

#[cfg(test)]
mod tests;
