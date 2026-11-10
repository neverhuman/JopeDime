//! Effort tiers: the deterministic work budget a comparator receives at each run effort.
//!
//! Comparators never run at `fast` effort, so there is no fast tier. A tier fixes work counts
//! (trees, rounds, candidates, ensemble rounds), which keeps results reproducible across machines.
//! The wall-clock ceiling only bounds the whole comparator stage; a comparator that hits it is
//! reported as timed out, never silently dropped.

use std::time::Duration;

/// `ComparatorTier` is an enum for the effort-matched comparator budget contract in this module:
/// the run effort a comparator's work counts and wall-clock ceiling are matched to.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ComparatorTier {
    Medium,
    High,
    Ultra,
}

/// `TierBudget` is a struct for the per-tier work count contract in this module.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TierBudget {
    /// Maximum boosting rounds per gradient-boosting model (early stopping may keep fewer).
    pub boosting_rounds: usize,
    /// Trees per standalone forest comparator.
    pub forest_trees: usize,
    /// Trees per forest member inside an AutoML-style portfolio.
    pub member_forest_trees: usize,
    /// Greedy ensemble-selection rounds for portfolio comparators.
    pub ensemble_rounds: usize,
    /// Configurations evaluated by search-style comparators (FLAML-, H2O-style).
    pub search_candidates: usize,
    /// Cross-validation folds used by search and stacking comparators.
    pub folds: usize,
    /// Wall-clock ceiling for the whole comparator stage at this tier.
    pub stage_ceiling: Duration,
}

impl ComparatorTier {
    /// Every tier, cheapest first.
    pub const ALL: [Self; 3] = [Self::Medium, Self::High, Self::Ultra];

    /// Stable lower-case name, matching the run effort it serves.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Medium => "medium",
            Self::High => "high",
            Self::Ultra => "ultra",
        }
    }

    /// Parse a run effort name; `fast` and unknown names have no tier.
    pub fn from_effort(effort: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|tier| tier.as_str() == effort)
    }

    /// The deterministic work counts and stage ceiling for this tier.
    pub fn budget(self) -> TierBudget {
        match self {
            Self::Medium => TierBudget {
                boosting_rounds: 200,
                forest_trees: 100,
                member_forest_trees: 50,
                ensemble_rounds: 35,
                search_candidates: 8,
                folds: 3,
                stage_ceiling: Duration::from_secs(60),
            },
            Self::High => TierBudget {
                boosting_rounds: 500,
                forest_trees: 300,
                member_forest_trees: 100,
                ensemble_rounds: 50,
                search_candidates: 24,
                folds: 5,
                stage_ceiling: Duration::from_secs(300),
            },
            Self::Ultra => TierBudget {
                boosting_rounds: 1_500,
                forest_trees: 500,
                member_forest_trees: 200,
                ensemble_rounds: 100,
                search_candidates: 64,
                folds: 5,
                stage_ceiling: Duration::from_secs(1_800),
            },
        }
    }
}
