//! Run-control kernel, spec §5.2/§5.6. Callers serialize mutations through `&mut`.
//!
//! SVC-01/02 must commit state/events/command receipts atomically and persist the
//! fencing high-water before dispatch. `recover_interrupted` models recovery of
//! that retained controller state; it is not a disk loader. SVC-04 owns publishing,
//! DSC-05/08 own admission budgets/discovery, and WRK-01 owns worker supervision.
//! No dataset rows, model fitting, store transactions or HTTP handlers live here.
//!
//! A pause deadline rolls back every uncommitted round, including buffered results.
//! Continue must redispatch those round/slot seeds with new attempts. Captured round
//! effort survives pause/crash rollback; later effort changes apply to future rounds.
//! Quiescent phase advances are checkpoint boundaries and survive recovery.

mod controller;
mod evaluator;
pub(crate) mod landmarks;
pub mod learned;
pub(crate) mod service;
pub(crate) mod surrogate;

pub use controller::*;
pub use evaluator::*;

#[cfg(test)]
mod learned_tests;
#[cfg(test)]
mod tests;
