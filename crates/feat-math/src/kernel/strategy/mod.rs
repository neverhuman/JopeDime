//! Versioned candidate policies and pure deterministic search control.

pub mod gumbel_root;
pub mod migration;
pub mod modern_gp;
pub mod qd;
pub mod race;
pub mod residual;
pub mod selection;

mod recorder;
mod state;

pub use recorder::{Event, EventKind, Recorder, Trace};
pub use state::{
    execute, execute_with_recorder, Allocator, Candidate, Cost, Evaluation, Evaluator, Identity,
    Outcome, Proposal, Proposer, Reservation, Run, Strategy, StrategyConfig, StrategyError,
    STRATEGY_EVENT_VERSION, STRATEGY_ID, STRATEGY_VERSION,
};
