//! Pipeline-side executor boundary for the four model-build routes (SC-DEC-184T item 4).
//!
//! The four routes are the `(automl, ensemble)` table in `config::route`. Each route starts with a
//! verified neural checkpoint, an admitted identity and an admitted budget. `admit_route` checks
//! all three before any stage runs, then reports the stages that have no executor.
//!
//! Three limits are stated here so they are not read as more than they are:
//! - A `VerifiedNeuralCheckpoint` is an identity match against a P02 closure. It is not a loaded
//!   runtime, and no runtime loader exists in this crate. Every neural route therefore stays
//!   `Pending` on the neural-to-GP stage, and no route substitutes a generic GP or ridge fit.
//! - The `VerifiedReleaseManifest` named in the review does not exist in this repository. Nothing
//!   here treats release metadata as a loaded checkpoint.
//! - The 80-versus-64 conflict is resolved by refusing: the neural initial population has 80
//!   slots, so a `low` request whose evaluation cap is 64 cannot run it and is refused at
//!   admission, before any stage starts.
use super::config::{route_of, Route, VerifiedNeuralCheckpoint};
use super::identity::ExecutionIdentity;
use super::selection::{BudgetKind, EvaluationBudget};
use crate::pathway::ThinkingLevel;
use anyhow::{ensure, Result};

/// Generation-zero slots of the neural search. Mirrors `neural_search::INITIAL_SIZE`, which sits
/// behind the `neural-search` feature, so it is repeated here as a plain constant.
pub const NEURAL_INITIAL_POPULATION: u64 = 80;
/// Neural slots within the initial population. Mirrors `neural_search::NEURAL_SLOTS`.
pub const NEURAL_SLOTS: u64 = 64;

/// One stage of a route, in execution order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Stage {
    NeuralToGp,
    NativeBakeoff,
    WinnerEnsemble,
}

/// The stages a route runs. Every route begins with the neural-to-GP stage.
pub fn stages(route: Route) -> Vec<Stage> {
    match route {
        Route::FastNeuralGp => vec![Stage::NeuralToGp],
        Route::NeuralGpAutomlBakeoff => vec![Stage::NeuralToGp, Stage::NativeBakeoff],
        Route::NeuralGpWinnerEnsemble => vec![Stage::NeuralToGp, Stage::WinnerEnsemble],
        Route::CombinedRosterWinnerEnsemble => vec![
            Stage::NeuralToGp,
            Stage::NativeBakeoff,
            Stage::WinnerEnsemble,
        ],
    }
}

/// Everything a route needs before it may start. Each field is a boundary, not a hint.
pub struct RouteBoundary<'a> {
    pub route: Route,
    pub level: ThinkingLevel,
    pub checkpoint: &'a VerifiedNeuralCheckpoint,
    pub identity: &'a ExecutionIdentity,
    pub budget: &'a mut EvaluationBudget,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RouteAdmission {
    /// Every stage has an executor and the boundaries are admitted.
    Admitted { stages: Vec<Stage> },
    /// The boundaries are admitted, but these stages have no executor yet. Nothing ran.
    Pending {
        stages: Vec<Stage>,
        /// The plan the route was admitted against.
        plan_sha256: String,
        /// The verified checkpoint identity the route was admitted against.
        checkpoint_sha256: String,
        missing: Vec<&'static str>,
    },
}

/// Admits a route. Refuses, before any stage runs, when the level cannot hold the neural initial
/// population or when the budget cannot pay for it. Charges the population as neural work.
pub fn admit_route(boundary: RouteBoundary<'_>) -> Result<RouteAdmission> {
    let RouteBoundary {
        route,
        level,
        checkpoint,
        identity,
        budget,
    } = boundary;
    ensure!(
        level.cap() >= NEURAL_INITIAL_POPULATION,
        "thinking level {level:?} caps evaluations at {} but the neural initial population has {} slots ({} neural); a low-budget request cannot run it",
        level.cap(),
        NEURAL_INITIAL_POPULATION,
        NEURAL_SLOTS
    );
    budget.check_wall_clock().map_err(|e| {
        anyhow::anyhow!("the route's wall-clock budget is exhausted before admission: {e:?}")
    })?;
    budget
        .charge(BudgetKind::NeuralGpWork, NEURAL_INITIAL_POPULATION)
        .map_err(|e| anyhow::anyhow!("neural initial population over the request budget: {e:?}"))?;

    let stages = stages(route);
    // The checkpoint is verified as an identity. Nothing loads its weights, so the neural stage
    // cannot run, whatever the route.
    Ok(RouteAdmission::Pending {
        stages,
        plan_sha256: identity.plan_sha256.clone(),
        checkpoint_sha256: checkpoint.checkpoint_sha256().into(),
        missing: vec![
            "loaded neural checkpoint runtime",
            "final neural-to-GP executor",
        ],
    })
}

/// The four independent `(automl, ensemble)` Boolean combinations, each admitted through the same
/// boundary as `admit_route` (SC-DEC-184V item 3).
pub fn admit_combination(
    automl: bool,
    ensemble: bool,
    level: ThinkingLevel,
    checkpoint: &VerifiedNeuralCheckpoint,
    identity: &ExecutionIdentity,
    budget: &mut EvaluationBudget,
) -> Result<RouteAdmission> {
    admit_route(RouteBoundary {
        route: route_of(automl, ensemble),
        level,
        checkpoint,
        identity,
        budget,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_route_starts_with_the_neural_stage() {
        for route in [
            Route::FastNeuralGp,
            Route::NeuralGpAutomlBakeoff,
            Route::NeuralGpWinnerEnsemble,
            Route::CombinedRosterWinnerEnsemble,
        ] {
            assert_eq!(stages(route)[0], Stage::NeuralToGp);
        }
    }

    #[test]
    fn the_combined_route_runs_the_bakeoff_then_the_ensemble() {
        assert_eq!(
            stages(Route::CombinedRosterWinnerEnsemble),
            vec![
                Stage::NeuralToGp,
                Stage::NativeBakeoff,
                Stage::WinnerEnsemble
            ]
        );
    }
}
