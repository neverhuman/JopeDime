//! Frozen lite-race control adapter.
//!
//! Rung row cuts, fold counts, head proxies, rung budgets and tie handling stay
//! in `evolve_lite_race`; this module adds only a versioned control identity.

use feat_math::{evolve, Rows};

/// Historical algorithm identity. Old benchmark observations retain the
/// method/source identities with which they were originally recorded.
pub const STRATEGY_ID: &str = "feat-math.lite-race.v1";

/// Run the original three-rung lite-race protocol. The caller's seed, admitted
/// seeds, population, generation count and time budget pass through unchanged.
pub fn run(
    x: &Rows,
    y: &[f64],
    metric: &(dyn Fn(&[f64], &[f64]) -> f64 + Sync),
    config: &evolve::EvolveConfig,
    observer: &dyn evolve::Observer,
) -> evolve::RaceOutcome {
    let mut frozen = config.clone();
    frozen.fitness = evolve::FitnessMode::LiteRace;
    frozen.fault_policy = evolve::FaultPolicy::Legacy;
    frozen.selection = evolve::ParentSelection::Tournament3;
    frozen.ridge_scoring = evolve::RidgeScoring::Refit;
    evolve::evolve_lite_race(x, y, metric, &frozen, observer)
}
