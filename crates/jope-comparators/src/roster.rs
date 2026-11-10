//! The comparator roster: stable identifiers, honest display labels and the single entry point
//! that fits any comparator at an effort tier.

use crate::automl::stack::StackedEnsemble;
use crate::automl::{self, search, AUTOGLUON_VALIDATION, MLJAR_VALIDATION};
use crate::deadline::Deadline;
use crate::error::{ComparatorError, ComparatorResult};
use crate::forest::ForestParams;
use crate::matrix::{validate_training, FeatureMatrix};
use crate::portfolio::{FittedMember, MemberSpec, WeightedPortfolio};
use crate::presets;
use crate::tier::ComparatorTier;

/// `ComparatorId` is an enum for the comparator roster contract in this module: the stable
/// identifiers of the pure-Rust comparators reported beside a run's champion.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum ComparatorId {
    AutogluonStyle,
    FlamlStyle,
    H2oStyle,
    MljarStyle,
    XgboostStyle,
    LightgbmStyle,
    CatboostStyle,
    RandomForest,
    ExtraTrees,
    HistGradientBoosting,
}

/// `ComparatorDescriptor` is a struct for the comparator display contract in this module: the
/// honest label, family and known gaps reported with every comparator result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ComparatorDescriptor {
    pub id: ComparatorId,
    /// Stable snake_case identifier used by the CLI, MCP and reports.
    pub key: &'static str,
    /// Display label; always says it is a Rust analogue or port.
    pub label: &'static str,
    /// The upstream tool whose strategy or defaults this comparator follows.
    pub reference: &'static str,
    /// What this comparator does not reproduce.
    pub known_gaps: &'static str,
}

const DESCRIPTORS: [ComparatorDescriptor; 10] = [
    ComparatorDescriptor {
        id: ComparatorId::AutogluonStyle,
        key: "autogluon_style",
        label: "AutoGluon-style weighted ensemble (Rust port)",
        reference: "AutoGluon Tabular medium_quality",
        known_gaps: "no neural-network members; six-member portfolio; single inner holdout, no bagging or multi-layer stacking",
    },
    ComparatorDescriptor {
        id: ComparatorId::FlamlStyle,
        key: "flaml_style",
        label: "FLAML-style cost-frugal search (Rust port)",
        reference: "FLAML AutoML",
        known_gaps: "CFO/BlendSearch simplified to a seeded local search with a fixed candidate count",
    },
    ComparatorDescriptor {
        id: ComparatorId::H2oStyle,
        key: "h2o_style",
        label: "H2O AutoML-style stacked ensemble (Rust port)",
        reference: "H2O AutoML",
        known_gaps: "no DeepLearning or native XGBoost; small seeded GBM grid; non-negative linear metalearner",
    },
    ComparatorDescriptor {
        id: ComparatorId::MljarStyle,
        key: "mljar_style",
        label: "MLJAR-style AutoML (Rust port)",
        reference: "mljar-supervised Perform mode",
        known_gaps: "no neural network or golden features; default parameters only; single inner holdout",
    },
    ComparatorDescriptor {
        id: ComparatorId::XgboostStyle,
        key: "xgboost_style",
        label: "XGBoost-style hist-GBM (Rust)",
        reference: "XGBoost hist defaults",
        known_gaps: "shared histogram learner with XGBoost-like settings, not XGBoost's split finding",
    },
    ComparatorDescriptor {
        id: ComparatorId::LightgbmStyle,
        key: "lightgbm_style",
        label: "LightGBM-style leaf-wise hist-GBM (Rust)",
        reference: "LightGBM gbdt defaults",
        known_gaps: "no GOSS or exclusive feature bundling",
    },
    ComparatorDescriptor {
        id: ComparatorId::CatboostStyle,
        key: "catboost_style",
        label: "CatBoost-style hist-GBM, strong L2 (Rust)",
        reference: "CatBoost defaults",
        known_gaps: "depth-wise trees; no oblivious trees, ordered boosting or categorical encodings",
    },
    ComparatorDescriptor {
        id: ComparatorId::RandomForest,
        key: "random_forest",
        label: "Random forest (scikit-learn defaults, Rust)",
        reference: "scikit-learn RandomForestRegressor",
        known_gaps: "255-bin histogram splits instead of exact thresholds",
    },
    ComparatorDescriptor {
        id: ComparatorId::ExtraTrees,
        key: "extra_trees",
        label: "Extra trees (scikit-learn defaults, Rust)",
        reference: "scikit-learn ExtraTreesRegressor",
        known_gaps: "random cuts drawn between occupied histogram bins",
    },
    ComparatorDescriptor {
        id: ComparatorId::HistGradientBoosting,
        key: "hist_gradient_boosting",
        label: "HistGradientBoosting (scikit-learn-style, Rust)",
        reference: "scikit-learn HistGradientBoostingRegressor",
        known_gaps: "squared loss only; no categorical or monotonic constraints",
    },
];

impl ComparatorId {
    /// Every comparator, in report order.
    pub const ALL: [Self; 10] = [
        Self::AutogluonStyle,
        Self::FlamlStyle,
        Self::H2oStyle,
        Self::MljarStyle,
        Self::XgboostStyle,
        Self::LightgbmStyle,
        Self::CatboostStyle,
        Self::RandomForest,
        Self::ExtraTrees,
        Self::HistGradientBoosting,
    ];

    /// The display and provenance record of this comparator.
    pub fn descriptor(self) -> &'static ComparatorDescriptor {
        &DESCRIPTORS[self as usize]
    }

    /// Stable snake_case identifier.
    pub fn key(self) -> &'static str {
        self.descriptor().key
    }

    /// Resolve a stable identifier; `None` for an unknown name.
    pub fn from_key(key: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|id| id.key() == key)
    }
}

/// `FittedComparator` is a struct for the fitted comparator contract in this module: any roster
/// entry after fitting, predicting through one interface.
#[derive(Clone, Debug, PartialEq)]
pub struct FittedComparator {
    id: ComparatorId,
    inner: FittedKind,
}

/// `FittedKind` is an enum for the fitted comparator shape contract in this module.
#[derive(Clone, Debug, PartialEq)]
enum FittedKind {
    Single(FittedMember),
    Portfolio(WeightedPortfolio),
    Stack(StackedEnsemble),
}

impl FittedComparator {
    /// The roster entry this model was fitted as.
    pub fn id(&self) -> ComparatorId {
        self.id
    }

    /// Predict each row; refuses a feature width different from the fit.
    pub fn predict(&self, features: FeatureMatrix<'_>) -> ComparatorResult<Vec<f64>> {
        match &self.inner {
            FittedKind::Single(model) => model.predict(features),
            FittedKind::Portfolio(model) => model.predict(features),
            FittedKind::Stack(model) => model.predict(features),
        }
    }

    /// Approximate in-memory parameter footprint in bytes, for model-size comparison.
    pub fn model_bytes(&self) -> u64 {
        match &self.inner {
            FittedKind::Single(model) => model.model_bytes(),
            FittedKind::Portfolio(model) => model.model_bytes(),
            FittedKind::Stack(model) => model.model_bytes(),
        }
    }

    /// Number of fitted models inside this comparator.
    pub fn model_count(&self) -> usize {
        match &self.inner {
            FittedKind::Single(_) => 1,
            FittedKind::Portfolio(model) => model.member_count(),
            FittedKind::Stack(model) => model.member_count(),
        }
    }
}

/// Fit comparator `id` at `tier` on training rows only.
pub fn fit_comparator(
    id: ComparatorId,
    tier: ComparatorTier,
    features: FeatureMatrix<'_>,
    targets: &[f64],
    seed: u64,
    deadline: &Deadline,
) -> ComparatorResult<FittedComparator> {
    validate_training(features, targets)?;
    if features.rows() < 2 {
        return Err(ComparatorError::EmptyFit);
    }
    let budget = tier.budget();
    let rounds = budget.boosting_rounds;
    let single = |spec: MemberSpec| -> ComparatorResult<FittedKind> {
        Ok(FittedKind::Single(
            spec.fit(features, targets, seed, deadline)?,
        ))
    };
    let inner = match id {
        ComparatorId::AutogluonStyle => FittedKind::Portfolio(WeightedPortfolio::fit(
            &automl::autogluon_members(&budget),
            features,
            targets,
            AUTOGLUON_VALIDATION,
            budget.ensemble_rounds,
            seed,
            deadline,
        )?),
        ComparatorId::MljarStyle => FittedKind::Portfolio(WeightedPortfolio::fit(
            &automl::mljar_members(&budget),
            features,
            targets,
            MLJAR_VALIDATION,
            budget.ensemble_rounds,
            seed,
            deadline,
        )?),
        ComparatorId::FlamlStyle => {
            FittedKind::Single(search::fit(features, targets, &budget, seed, deadline)?)
        }
        ComparatorId::H2oStyle => FittedKind::Stack(StackedEnsemble::fit(
            features, targets, &budget, seed, deadline,
        )?),
        ComparatorId::XgboostStyle => single(MemberSpec::Boosting(presets::xgboost_style(rounds)))?,
        ComparatorId::LightgbmStyle => {
            single(MemberSpec::Boosting(presets::lightgbm_style(rounds)))?
        }
        ComparatorId::CatboostStyle => {
            single(MemberSpec::Boosting(presets::catboost_style(rounds)))?
        }
        ComparatorId::HistGradientBoosting => {
            single(MemberSpec::Boosting(presets::scikit_learn_style(rounds)))?
        }
        ComparatorId::RandomForest => single(MemberSpec::Forest(ForestParams::random_forest(
            budget.forest_trees,
        )))?,
        ComparatorId::ExtraTrees => single(MemberSpec::Forest(ForestParams::extra_trees(
            budget.forest_trees,
        )))?,
    };
    Ok(FittedComparator { id, inner })
}
