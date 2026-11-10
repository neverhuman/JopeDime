//! FLAML-style cost-frugal search: start from cheap configurations and walk outward from the
//! incumbent, scoring every candidate by k-fold cross-validated RMSE on the training rows.
//!
//! FLAML's CFO/BlendSearch is simplified to a seeded local search with a fixed candidate count,
//! so results are reproducible and independent of machine speed.

use crate::deadline::Deadline;
use crate::error::{ComparatorError, ComparatorResult};
use crate::forest::ForestParams;
use crate::histgbm::HistGbmParams;
use crate::matrix::FeatureMatrix;
use crate::metrics::rmse;
use crate::portfolio::{fold_assignment, out_of_fold, FittedMember, MemberSpec};
use crate::presets;
use crate::rng::{stream_seed, SplitMix64};
use crate::tier::TierBudget;
use std::collections::VecDeque;

/// FLAML's low-cost starting points: a 4-leaf GBM, a small forest and small extra trees.
fn starting_points(budget: &TierBudget) -> Vec<MemberSpec> {
    let small_trees = (budget.member_forest_trees / 2).max(4);
    vec![
        MemberSpec::Boosting(HistGbmParams {
            max_leaves: Some(4),
            learning_rate: 0.1,
            min_samples_leaf: 20,
            ..presets::lightgbm_style(budget.boosting_rounds)
        }),
        MemberSpec::Forest(ForestParams::random_forest(small_trees)),
        MemberSpec::Forest(ForestParams::extra_trees(small_trees)),
    ]
}

/// One seeded step away from `incumbent`, growing or shrinking its capacity.
fn neighbour(incumbent: &MemberSpec, budget: &TierBudget, rng: &mut SplitMix64) -> MemberSpec {
    let grow = rng.below(2) == 0;
    match incumbent {
        MemberSpec::Boosting(params) => {
            let mut next = params.clone();
            match rng.below(3) {
                0 => {
                    let leaves = next.max_leaves.unwrap_or(31);
                    next.max_leaves = Some(if grow {
                        leaves * 2
                    } else {
                        (leaves / 2).max(2)
                    });
                }
                1 => {
                    let rate = if grow {
                        next.learning_rate * 1.5
                    } else {
                        next.learning_rate / 1.5
                    };
                    next.learning_rate = rate.clamp(0.005, 1.0);
                }
                _ => {
                    let leaf = next.min_samples_leaf;
                    next.min_samples_leaf = if grow { (leaf / 2).max(1) } else { leaf * 2 };
                }
            }
            MemberSpec::Boosting(next)
        }
        MemberSpec::Forest(params) => {
            let mut next = params.clone();
            if grow {
                next.trees = (next.trees * 2).min(budget.forest_trees);
            } else {
                next.max_features = (next.max_features * 0.7).max(0.1);
            }
            MemberSpec::Forest(next)
        }
        MemberSpec::Ridge(params) => MemberSpec::Ridge(params.clone()),
    }
}

/// Run the search and refit the best configuration on every training row.
pub(crate) fn fit(
    features: FeatureMatrix<'_>,
    targets: &[f64],
    budget: &TierBudget,
    seed: u64,
    deadline: &Deadline,
) -> ComparatorResult<FittedMember> {
    let folds = fold_assignment(features.rows(), budget.folds, stream_seed(seed, 0));
    let mut rng = SplitMix64::new(stream_seed(seed, 1));
    let mut queue: VecDeque<MemberSpec> = starting_points(budget).into();
    let mut best: Option<(f64, MemberSpec)> = None;
    for candidate in 0..budget.search_candidates.max(1) {
        let spec = match queue.pop_front() {
            Some(spec) => spec,
            None => match &best {
                Some((_, incumbent)) => neighbour(incumbent, budget, &mut rng),
                None => break,
            },
        };
        let predictions = out_of_fold(
            &spec,
            features,
            targets,
            &folds,
            stream_seed(seed, candidate as u64 + 2),
            deadline,
        )?;
        let score = rmse(targets, &predictions)?;
        if best.as_ref().is_none_or(|(current, _)| score < *current) {
            best = Some((score, spec));
        }
    }
    let (_, spec) = best.ok_or(ComparatorError::InvalidParameter {
        name: "search_candidates",
    })?;
    spec.fit(features, targets, stream_seed(seed, 1_000_003), deadline)
}
