//! AutoML-style comparators built on the shared portfolio core.
//!
//! - **AutoGluon-style** (`medium_quality` strategy): six diverse members fitted on a 20 % inner
//!   holdout, then a Caruana weighted ensemble (`WeightedEnsemble_L2`), refitted on all rows.
//! - **MLJAR-style** (`Perform` mode strategy): seven default-parameter members on a 25 % inner
//!   holdout, then greedy ensembling, refitted on all rows.
//! - **FLAML-style**: cost-frugal configuration search ([`search`]).
//! - **H2O AutoML-style**: out-of-fold stacked ensemble ([`stack`]).
//!
//! These reproduce each tool's *strategy* on the shared Rust learners. They omit neural-network
//! members and use histogram GBM analogues for XGBoost, LightGBM and CatBoost.

pub(crate) mod search;
pub(crate) mod stack;

use crate::forest::ForestParams;
use crate::linear::RidgeParams;
use crate::portfolio::MemberSpec;
use crate::presets;
use crate::tier::TierBudget;

/// Inner validation fraction of the AutoGluon-style portfolio.
pub(crate) const AUTOGLUON_VALIDATION: f64 = 0.2;
/// Inner validation fraction of the MLJAR-style portfolio.
pub(crate) const MLJAR_VALIDATION: f64 = 0.25;

/// AutoGluon `medium_quality` regression portfolio, in AutoGluon's member order.
pub(crate) fn autogluon_members(budget: &TierBudget) -> Vec<MemberSpec> {
    vec![
        MemberSpec::Boosting(presets::lightgbm_style(budget.boosting_rounds)),
        MemberSpec::Boosting(presets::xgboost_style(budget.boosting_rounds)),
        MemberSpec::Forest(ForestParams::random_forest(budget.member_forest_trees)),
        MemberSpec::Forest(ForestParams::extra_trees(budget.member_forest_trees)),
        MemberSpec::Boosting(presets::catboost_style(budget.boosting_rounds)),
        MemberSpec::Ridge(RidgeParams { alpha: 1.0 }),
    ]
}

/// MLJAR `Perform` mode algorithm list with default parameters (neural network omitted).
pub(crate) fn mljar_members(budget: &TierBudget) -> Vec<MemberSpec> {
    vec![
        MemberSpec::Ridge(RidgeParams { alpha: 1e-3 }),
        MemberSpec::Forest(ForestParams::random_forest(budget.member_forest_trees)),
        MemberSpec::Forest(ForestParams::extra_trees(budget.member_forest_trees)),
        MemberSpec::Boosting(presets::xgboost_style(budget.boosting_rounds)),
        MemberSpec::Boosting(presets::lightgbm_style(budget.boosting_rounds)),
        MemberSpec::Boosting(presets::catboost_style(budget.boosting_rounds)),
        MemberSpec::Boosting(presets::scikit_learn_style(budget.boosting_rounds)),
    ]
}
