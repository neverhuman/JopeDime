//! Isolated best-five DL-seeded GP regression campaign.
//!
//! This protocol owns no production pointers and deliberately does not reuse
//! mutable state or rankings from `jope_strategy_screen`, Search V3, or the
//! generalized GP campaign.  Immutable predictions may be shared through the
//! fully-bound cache key only.

mod budget_campaign;
mod compare;
mod contracts;
mod execution;
mod portfolio;
mod scq;
mod splits;
mod strategies;
mod worker;

pub use budget_campaign::*;
pub use compare::{compare_campaign, deterministic_lower_bound};
pub use contracts::*;
pub use execution::*;
pub use portfolio::*;
pub use scq::{
    build_scq_plan, emit_and_check_plan, render_scq_spec, render_scq_spec_with_io,
    validate_pilot_selection_coverage, verify_submission_weight, ScqInputV1, ScqJobV1, ScqOutputV1,
    ScqPlanV1, ScqStageV1,
};
pub use splits::{build_cv7_plan, build_search_split_plan};
pub use strategies::*;
pub use worker::*;
