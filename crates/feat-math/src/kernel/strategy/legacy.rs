//! Frozen legacy search control adapter.
//!
//! This intentionally delegates to the existing public evolution loop. It does
//! not reinterpret seeds, draw random values, reorder slots, or normalize results.
//! The adapter is a stable identity boundary for I17 drivers while that shared
//! strategy API is integrated.

use feat_math::{evolve, Rows};

/// Historical algorithm identity. Existing receipts keep their own source and
/// profile identities; this name does not retroactively qualify them.
pub const STRATEGY_ID: &str = "feat-math.legacy-gp.v1";

/// Run the frozen surrogate control with the caller's original configuration.
/// Only the fitness selector is pinned to the historical surrogate loop; seed
/// list order, seed value, population, generation count and limits are retained.
pub fn run(
    x: &Rows,
    y: &[f64],
    metric: &(dyn Fn(&[f64], &[f64]) -> f64 + Sync),
    config: &evolve::EvolveConfig,
    observer: &dyn evolve::Observer,
) -> Vec<evolve::Scored> {
    let mut frozen = config.clone();
    frozen.fitness = evolve::FitnessMode::Surrogate;
    frozen.fault_policy = evolve::FaultPolicy::Legacy;
    frozen.selection = evolve::ParentSelection::Tournament3;
    frozen.ridge_scoring = evolve::RidgeScoring::Refit;
    evolve::evolve(x, y, metric, &frozen, observer)
}
