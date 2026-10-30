//! The rung-0 elite set from spec §5.5. User objectives decide membership;
//! surrogate scores only order the retained members for revealed context.

use serde::{Deserialize, Deserializer, Serialize};
use std::cmp::Ordering;
use std::collections::BTreeSet;
use std::fmt;

pub const ELITE_CAPACITY: usize = 48;

/// Score directions are explicit: a maximized user objective must not be evicted
/// using a minimization comparison. Neither score is transformed by this pool.
#[derive(Clone, Copy, Debug, Deserialize, Eq, PartialEq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum ScoreOrder {
    Minimize,
    Maximize,
}

impl ScoreOrder {
    fn compare(self, left: f64, right: f64) -> Ordering {
        // All stored scores have passed finite validation. Numeric comparison
        // treats -0.0 and +0.0 as ties, resolved by algorithm ID below.
        match self {
            Self::Minimize => left.partial_cmp(&right),
            Self::Maximize => right.partial_cmp(&left),
        }
        .expect("elite scores are finite")
    }
}

/// Measurements for one algorithm on the run's rung-0 folds. IDs refer to the
/// run's algorithm store; the pool does not own programs or evaluate objectives.
#[derive(Clone, Debug, Deserialize, PartialEq, Serialize)]
#[serde(deny_unknown_fields)]
pub struct Elite {
    pub algorithm_id: String,
    pub surrogate: f64,
    pub objective: f64,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InvalidScore {
    Surrogate,
    Objective,
}

impl fmt::Display for InvalidScore {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "non-finite {} elite score",
            match self {
                Self::Surrogate => "surrogate",
                Self::Objective => "objective",
            }
        )
    }
}

impl std::error::Error for InvalidScore {}

impl Elite {
    fn validate(&self) -> Result<(), InvalidScore> {
        if !self.surrogate.is_finite() {
            return Err(InvalidScore::Surrogate);
        }
        if !self.objective.is_finite() {
            return Err(InvalidScore::Objective);
        }
        Ok(())
    }
}

/// One set, stored best-first by objective with ascending algorithm-ID ties.
/// Serialization retains both directions and a canonical membership ordering.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct ElitePool {
    surrogate_order: ScoreOrder,
    objective_order: ScoreOrder,
    entries: Vec<Elite>,
}

impl ElitePool {
    pub fn new(surrogate_order: ScoreOrder, objective_order: ScoreOrder) -> Self {
        Self {
            surrogate_order,
            objective_order,
            entries: Vec::new(),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Replace measurements for an existing ID, or insert a new ID. Above 48,
    /// return the worst objective member evicted (possibly the incoming member).
    /// Invalid measurements leave the pool unchanged.
    pub fn insert(&mut self, elite: Elite) -> Result<Option<Elite>, InvalidScore> {
        elite.validate()?;
        self.entries
            .retain(|stored| stored.algorithm_id != elite.algorithm_id);
        self.entries.push(elite);
        self.sort_by_objective();
        Ok((self.entries.len() > ELITE_CAPACITY)
            .then(|| self.entries.pop().expect("pool exceeds capacity")))
    }

    /// Best-first objective order over every retained member.
    pub fn by_objective(&self) -> &[Elite] {
        &self.entries
    }

    /// Best surrogate members from this same set, capped by its size. Discovery
    /// requests four for the revealed context; this never changes membership.
    pub fn top_k_by_surrogate(&self, k: usize) -> Vec<&Elite> {
        let mut ordered: Vec<_> = self.entries.iter().collect();
        ordered.sort_by(|a, b| {
            self.surrogate_order
                .compare(a.surrogate, b.surrogate)
                .then_with(|| a.algorithm_id.cmp(&b.algorithm_id))
        });
        ordered.truncate(k);
        ordered
    }

    /// Zero-based rank; `rank < 4` is the spec's top-four promotion condition.
    /// Unknown or evicted IDs have no rank.
    pub fn rank_by_objective(&self, id: &str) -> Option<usize> {
        self.entries
            .iter()
            .position(|entry| entry.algorithm_id == id)
    }

    fn sort_by_objective(&mut self) {
        let order = self.objective_order;
        self.entries.sort_by(|a, b| {
            order
                .compare(a.objective, b.objective)
                .then_with(|| a.algorithm_id.cmp(&b.algorithm_id))
        });
    }
}

impl<'de> Deserialize<'de> for ElitePool {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Stored {
            surrogate_order: ScoreOrder,
            objective_order: ScoreOrder,
            entries: Vec<Elite>,
        }

        let stored = Stored::deserialize(deserializer)?;
        if stored.entries.len() > ELITE_CAPACITY {
            return Err(serde::de::Error::custom("elite pool exceeds capacity 48"));
        }
        let mut ids = BTreeSet::new();
        for entry in &stored.entries {
            entry.validate().map_err(serde::de::Error::custom)?;
            if !ids.insert(&entry.algorithm_id) {
                return Err(serde::de::Error::custom("duplicate elite algorithm ID"));
            }
        }
        let mut pool = Self {
            surrogate_order: stored.surrogate_order,
            objective_order: stored.objective_order,
            entries: stored.entries,
        };
        pool.sort_by_objective();
        Ok(pool)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(id: usize, surrogate: f64, objective: f64) -> Elite {
        Elite {
            algorithm_id: format!("algo-{id:03}"),
            surrogate,
            objective,
        }
    }

    fn pool() -> ElitePool {
        ElitePool::new(ScoreOrder::Minimize, ScoreOrder::Minimize)
    }

    fn ids(entries: impl IntoIterator<Item = impl AsRef<str>>) -> Vec<String> {
        entries
            .into_iter()
            .map(|id| id.as_ref().to_owned())
            .collect()
    }

    #[test]
    fn empty_pool_and_top_k_boundaries() {
        let mut pool = pool();
        assert!(pool.is_empty());
        assert_eq!(pool.len(), 0);
        assert!(pool.top_k_by_surrogate(4).is_empty());
        assert_eq!(pool.rank_by_objective("missing"), None);
        pool.insert(entry(1, 1.0, 1.0)).unwrap();
        assert!(pool.top_k_by_surrogate(0).is_empty());
        assert_eq!(pool.top_k_by_surrogate(usize::MAX).len(), 1);
        assert_eq!(pool.rank_by_objective("algo-001"), Some(0));
    }

    #[test]
    fn capacity_evicts_by_user_objective_even_for_the_best_surrogate() {
        let mut pool = pool();
        for id in 0..48 {
            assert_eq!(pool.insert(entry(id, id as f64, id as f64)).unwrap(), None);
        }
        let rejected = entry(48, -100.0, 48.0);
        assert_eq!(pool.insert(rejected.clone()).unwrap(), Some(rejected));
        let evicted = pool.insert(entry(49, 100.0, -1.0)).unwrap().unwrap();
        assert_eq!(evicted.algorithm_id, "algo-047");
        assert_eq!(pool.len(), ELITE_CAPACITY);
        assert_eq!(pool.rank_by_objective("algo-049"), Some(0));
        assert_eq!(pool.rank_by_objective("algo-047"), None);
        assert_eq!(pool.top_k_by_surrogate(4)[0].algorithm_id, "algo-000");
    }

    #[test]
    fn surrogate_and_objective_orders_use_the_same_members() {
        let mut pool = pool();
        for id in 0..6 {
            pool.insert(entry(id, (5 - id) as f64, id as f64)).unwrap();
        }
        assert_eq!(
            ids(pool.top_k_by_surrogate(4).iter().map(|e| &e.algorithm_id)),
            ids(["algo-005", "algo-004", "algo-003", "algo-002"])
        );
        for (rank, e) in pool.by_objective().iter().enumerate() {
            assert_eq!(e.algorithm_id, format!("algo-{rank:03}"));
            assert_eq!(pool.rank_by_objective(&e.algorithm_id), Some(rank));
        }
        let surrogate_ids: BTreeSet<_> = pool
            .top_k_by_surrogate(48)
            .iter()
            .map(|e| e.algorithm_id.clone())
            .collect();
        assert_eq!(
            surrogate_ids,
            pool.by_objective()
                .iter()
                .map(|e| e.algorithm_id.clone())
                .collect()
        );
    }

    #[test]
    fn directions_are_independent_and_maximized_objective_evicts_the_lowest() {
        for surrogate in [ScoreOrder::Minimize, ScoreOrder::Maximize] {
            for objective in [ScoreOrder::Minimize, ScoreOrder::Maximize] {
                let mut pool = ElitePool::new(surrogate, objective);
                for id in (0..49).rev() {
                    pool.insert(entry(id, id as f64, id as f64)).unwrap();
                }
                let best = if objective == ScoreOrder::Minimize {
                    0
                } else {
                    48
                };
                let evicted = if objective == ScoreOrder::Minimize {
                    48
                } else {
                    0
                };
                assert_eq!(pool.rank_by_objective(&format!("algo-{best:03}")), Some(0));
                assert_eq!(pool.rank_by_objective(&format!("algo-{evicted:03}")), None);
                let first_surrogate = match (surrogate, objective) {
                    (ScoreOrder::Minimize, ScoreOrder::Minimize) => 0,
                    (ScoreOrder::Minimize, ScoreOrder::Maximize) => 1,
                    (ScoreOrder::Maximize, ScoreOrder::Minimize) => 47,
                    (ScoreOrder::Maximize, ScoreOrder::Maximize) => 48,
                };
                assert_eq!(
                    pool.top_k_by_surrogate(4)[0].algorithm_id,
                    format!("algo-{first_surrogate:03}")
                );
            }
        }
    }

    #[test]
    fn ties_and_capacity_are_id_ordered_independent_of_insertion_order() {
        let mut forward = pool();
        let mut reverse = pool();
        for id in 0..60 {
            forward.insert(entry(id, 0.0, -0.0)).unwrap();
        }
        for id in (0..60).rev() {
            reverse.insert(entry(id, -0.0, 0.0)).unwrap();
        }
        let expected: Vec<_> = (0..48).map(|id| format!("algo-{id:03}")).collect();
        for pool in [&forward, &reverse] {
            assert_eq!(
                ids(pool.by_objective().iter().map(|e| &e.algorithm_id)),
                expected
            );
            assert_eq!(
                ids(pool.top_k_by_surrogate(4).iter().map(|e| &e.algorithm_id)),
                expected[..4]
            );
        }
    }

    #[test]
    fn replacing_an_id_updates_both_orders_without_duplicate_membership() {
        let mut pool = pool();
        pool.insert(entry(1, 0.0, 5.0)).unwrap();
        pool.insert(entry(2, 1.0, 1.0)).unwrap();
        assert_eq!(pool.insert(entry(1, 2.0, 0.0)).unwrap(), None);
        assert_eq!(pool.len(), 2);
        assert_eq!(pool.rank_by_objective("algo-001"), Some(0));
        assert_eq!(pool.top_k_by_surrogate(1)[0].algorithm_id, "algo-002");
    }

    #[test]
    fn non_finite_scores_are_refused_without_mutating_the_pool() {
        let mut pool = pool();
        pool.insert(entry(1, 1.0, 1.0)).unwrap();
        let before = pool.clone();
        for score in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                pool.insert(entry(1, score, 0.0)),
                Err(InvalidScore::Surrogate)
            );
            assert_eq!(
                pool.insert(entry(2, 0.0, score)),
                Err(InvalidScore::Objective)
            );
            assert_eq!(pool, before);
        }
    }

    #[test]
    fn serde_round_trip_preserves_membership_directions_and_both_orders() {
        let mut pool = ElitePool::new(ScoreOrder::Maximize, ScoreOrder::Minimize);
        for id in (0..60).rev() {
            pool.insert(entry(id, id as f64, id as f64)).unwrap();
        }
        let json = serde_json::to_vec(&pool).unwrap();
        let decoded: ElitePool = serde_json::from_slice(&json).unwrap();
        assert_eq!(decoded, pool);
        assert_eq!(decoded.top_k_by_surrogate(4), pool.top_k_by_surrogate(4));
        assert_eq!(decoded.rank_by_objective("algo-047"), Some(47));
        let mut value = serde_json::to_value(&pool).unwrap();
        value["entries"].as_array_mut().unwrap().reverse();
        assert_eq!(serde_json::from_value::<ElitePool>(value).unwrap(), pool);
        let mut forward = ElitePool::new(ScoreOrder::Maximize, ScoreOrder::Minimize);
        for id in 0..60 {
            forward.insert(entry(id, id as f64, id as f64)).unwrap();
        }
        assert_eq!(serde_json::to_vec(&forward).unwrap(), json);
    }

    #[test]
    fn serde_refuses_corrupt_membership_instead_of_silently_repairing_it() {
        let mut pool = pool();
        pool.insert(entry(1, 1.0, 1.0)).unwrap();
        let valid = serde_json::to_value(&pool).unwrap();
        let mut duplicate = valid.clone();
        duplicate["entries"]
            .as_array_mut()
            .unwrap()
            .push(serde_json::to_value(entry(1, 2.0, 2.0)).unwrap());
        assert!(serde_json::from_value::<ElitePool>(duplicate).is_err());
        let mut oversized = valid.clone();
        oversized["entries"] =
            serde_json::to_value((0..49).map(|id| entry(id, 1.0, 1.0)).collect::<Vec<_>>())
                .unwrap();
        assert!(serde_json::from_value::<ElitePool>(oversized).is_err());
        let mut undefined = valid.clone();
        undefined["entries"][0]["objective"] = serde_json::Value::Null;
        assert!(serde_json::from_value::<ElitePool>(undefined).is_err());
        let mut unknown = valid;
        unknown["unexpected"] = true.into();
        assert!(serde_json::from_value::<ElitePool>(unknown).is_err());
    }
}
