//! Pure warm-start seeding for a discovery run (spec §5.7).

use super::elites::{Elite, ElitePool};
use crate::public::project::ProblemDigest;
use jope_api::error::ErrorCode;
use std::fmt;

/// The copied parent state used to seed a new discovery run.
#[derive(Clone, Debug, PartialEq)]
pub struct WarmStart {
    pub elites: ElitePool,
    pub champion: Elite,
}

/// A warm start can only reuse scores from the identical problem and a pool
/// containing a champion.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WarmStartError {
    pub code: ErrorCode,
    pub reason: &'static str,
}

impl fmt::Display for WarmStartError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{}: {}", self.code, self.reason)
    }
}

impl std::error::Error for WarmStartError {}

/// Copy the parent's elites and best objective member into a new run.
///
/// Scores are reusable only when both runs identify the same problem. The
/// pool's canonical objective ordering defines its champion.
pub fn seed_from_parent(
    parent_problem: ProblemDigest,
    new_problem: ProblemDigest,
    parent_elites: &ElitePool,
) -> Result<WarmStart, WarmStartError> {
    if parent_problem != new_problem {
        return Err(WarmStartError {
            code: ErrorCode::InvalidInput,
            reason: "warm start requires the same ProblemDigest as the parent",
        });
    }

    let champion = parent_elites
        .by_objective()
        .first()
        .cloned()
        .ok_or(WarmStartError {
            code: ErrorCode::InvalidInput,
            reason: "warm start requires a parent elite champion",
        })?;

    Ok(WarmStart {
        elites: parent_elites.clone(),
        champion,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discover::elites::ScoreOrder;

    fn parent_pool() -> ElitePool {
        let mut pool = ElitePool::new(ScoreOrder::Minimize, ScoreOrder::Minimize);
        pool.insert(Elite {
            algorithm_id: "alg-better".into(),
            surrogate: 1.0,
            objective: 2.0,
        })
        .unwrap();
        pool.insert(Elite {
            algorithm_id: "alg-worse".into(),
            surrogate: 2.0,
            objective: 5.0,
        })
        .unwrap();
        pool
    }

    fn digest(value: u8) -> ProblemDigest {
        ProblemDigest([value; 32])
    }

    #[test]
    fn warm_start_carries_parent_elites_and_champion_no_worse_than_parent() {
        let parent = parent_pool();
        let warm = seed_from_parent(digest(1), digest(1), &parent).unwrap();

        assert_eq!(warm.elites, parent);
        assert_eq!(warm.champion, parent.by_objective()[0]);
        assert!(warm.champion.objective <= parent.by_objective()[0].objective);
    }

    #[test]
    fn changed_problem_is_refused_as_invalid_input_with_reason() {
        let error = seed_from_parent(digest(1), digest(2), &parent_pool()).unwrap_err();

        assert_eq!(error.code, ErrorCode::InvalidInput);
        assert!(error.reason.contains("same ProblemDigest"));
    }
}
