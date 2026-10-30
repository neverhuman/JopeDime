//! Plain-language names and purposes for tunable model parameters.

use super::PublicName;

/// Public descriptions for tunable fields in planned operations, targets,
/// model heads, and search configuration.
pub const PARAMETERS: &[PublicName] = &[
    PublicName {
        key: "alpha",
        public_name: "regularization strength",
        purpose: "control how strongly model coefficients are kept small",
    },
    PublicName {
        key: "l1_ratio",
        public_name: "sparsity balance",
        purpose: "balance coefficient selection with coefficient shrinkage",
    },
    PublicName {
        key: "sweeps",
        public_name: "coordinate passes",
        purpose: "set how many passes are used to fit the model",
    },
    PublicName {
        key: "lambda_frac",
        public_name: "ridge penalty fraction",
        purpose: "set the strength of the ridge penalty",
    },
    PublicName {
        key: "rounds",
        public_name: "boosting rounds",
        purpose: "set how many successive trees are fitted",
    },
    PublicName {
        key: "depth",
        public_name: "tree depth",
        purpose: "limit how many levels each tree can use",
    },
    PublicName {
        key: "lr",
        public_name: "learning rate",
        purpose: "set how much each fitted tree changes the prediction",
    },
    PublicName {
        key: "lambda",
        public_name: "coefficient penalty",
        purpose: "limit the size of tree leaf values",
    },
    PublicName {
        key: "bins",
        public_name: "split search bins",
        purpose: "set how many candidate cut points are considered",
    },
    PublicName {
        key: "min_leaf",
        public_name: "minimum leaf size",
        purpose: "require each tree leaf to contain enough rows",
    },
    PublicName {
        key: "max_features",
        public_name: "feature limit",
        purpose: "cap the number of features considered by a tree",
    },
    PublicName {
        key: "leaves",
        public_name: "leaf limit",
        purpose: "set the maximum number of leaves in each tree",
    },
    PublicName {
        key: "bags",
        public_name: "tree bag count",
        purpose: "set how many independently sampled tree groups are fitted",
    },
    PublicName {
        key: "row_subsample",
        public_name: "row sampling share",
        purpose: "set the share of training rows used for each tree",
    },
    PublicName {
        key: "col_subsample",
        public_name: "feature sampling share",
        purpose: "set the share of features considered for each tree",
    },
    PublicName {
        key: "val_pct",
        public_name: "validation share",
        purpose: "set the share of training rows held aside for early stopping",
    },
    PublicName {
        key: "val_shuffle",
        public_name: "validation shuffle setting",
        purpose: "choose whether rows are shuffled before validation is selected",
    },
    PublicName {
        key: "huber_delta",
        public_name: "robust loss cutoff",
        purpose: "set where the fitting loss changes from squared to robust error",
    },
    PublicName {
        key: "stop_mae",
        public_name: "early stopping metric",
        purpose: "choose whether early stopping tracks mean absolute error",
    },
    PublicName {
        key: "m",
        public_name: "landmark count",
        purpose: "set how many training rows represent the kernel model",
    },
    PublicName {
        key: "steps",
        public_name: "kernel fitting steps",
        purpose: "set how many optimization steps fit the kernel model",
    },
    PublicName {
        key: "kernel",
        public_name: "kernel structure",
        purpose: "choose how similarity between rows is calculated",
    },
    PublicName {
        key: "k",
        public_name: "neighbor count",
        purpose: "set how many nearby training rows contribute to a prediction",
    },
    PublicName {
        key: "gamma",
        public_name: "kernel width",
        purpose: "set how quickly the influence of nearby rows decreases",
    },
    PublicName {
        key: "folds",
        public_name: "validation fold count",
        purpose: "set how many data partitions are used for validation",
    },
    PublicName {
        key: "repeats_auto",
        public_name: "automatic validation repeats",
        purpose: "allow the number of validation repeats to follow the data size",
    },
    PublicName {
        key: "q",
        public_name: "tail clipping fraction",
        purpose: "set how much of each target tail is clipped",
    },
    PublicName {
        key: "pop",
        public_name: "candidate count",
        purpose: "set how many candidate models are considered in each round",
    },
    PublicName {
        key: "gens",
        public_name: "search round count",
        purpose: "set how many rounds the search runs",
    },
    PublicName {
        key: "seed",
        public_name: "repeatability seed",
        purpose: "make randomized search choices repeatable",
    },
    PublicName {
        key: "max_transforms",
        public_name: "feature stage limit",
        purpose: "cap the number of feature processing stages before a model",
    },
    PublicName {
        key: "time_budget_secs",
        public_name: "time budget",
        purpose: "limit how long the search runs",
    },
    PublicName {
        key: "fitness",
        public_name: "scoring mode",
        purpose: "choose how candidate models are evaluated",
    },
    PublicName {
        key: "target_score",
        public_name: "early stopping score",
        purpose: "stop the search when a candidate reaches the configured score",
    },
    PublicName {
        key: "rank_robust",
        public_name: "stability-aware ranking",
        purpose: "include fold-to-fold score variation when ranking candidates",
    },
    PublicName {
        key: "hard_ten_priors",
        public_name: "challenging starter set",
        purpose: "include high-regularization and simple baseline starting models",
    },
    PublicName {
        key: "expansion_vocab",
        public_name: "expanded transform choices",
        purpose: "include the expanded set of feature transformations in search",
    },
    PublicName {
        key: "fault_policy",
        public_name: "candidate fault policy",
        purpose: "choose how invalid transformations affect candidate scoring",
    },
    PublicName {
        key: "selection",
        public_name: "candidate selection method",
        purpose: "choose how candidates are selected for the next round",
    },
    PublicName {
        key: "ridge_scoring",
        public_name: "ridge scoring method",
        purpose: "choose how ridge surrogate scores are computed",
    },
    // PlannedOp parameters use variant-qualified keys. Fields such as `k`,
    // `gamma`, and `seed` have different meanings across transforms, so their
    // public descriptions must remain tied to the operation that consumes them.
    PublicName {
        key: "op.quantile_bins.bins",
        public_name: "quantile bin count",
        purpose: "set how many value groups are formed from the training data",
    },
    PublicName {
        key: "op.bin_onehot.bins",
        public_name: "one-hot bin count",
        purpose: "set how many quantile groups receive separate output columns",
    },
    PublicName {
        key: "op.hinge.knots",
        public_name: "hinge knot count",
        purpose: "set how many data-derived breakpoints form the piecewise curves",
    },
    PublicName {
        key: "op.winsor_clip.q",
        public_name: "feature tail clip fraction",
        purpose: "set the fraction clipped from each feature tail",
    },
    PublicName {
        key: "op.cheb.degree",
        public_name: "polynomial degree",
        purpose: "set the highest polynomial order used for each feature",
    },
    PublicName {
        key: "op.rff_cos.n",
        public_name: "cosine feature count",
        purpose: "set how many cosine features approximate the smooth similarity",
    },
    PublicName {
        key: "op.rff_cos.gamma",
        public_name: "cosine similarity width",
        purpose: "set how quickly similarity decreases with distance",
    },
    PublicName {
        key: "op.rff_cos.seed",
        public_name: "cosine feature seed",
        purpose: "make randomized cosine features repeatable",
    },
    PublicName {
        key: "op.rff_sin.n",
        public_name: "sine feature count",
        purpose: "set how many sine features approximate the smooth similarity",
    },
    PublicName {
        key: "op.rff_sin.gamma",
        public_name: "sine similarity width",
        purpose: "set how quickly similarity decreases with distance",
    },
    PublicName {
        key: "op.rff_sin.seed",
        public_name: "sine feature seed",
        purpose: "make randomized sine features repeatable",
    },
    PublicName {
        key: "op.rand_proj.k",
        public_name: "projection count",
        purpose: "set how many projected feature directions are retained",
    },
    PublicName {
        key: "op.rand_proj.seed",
        public_name: "projection seed",
        purpose: "make the randomized projection repeatable",
    },
    PublicName {
        key: "op.landmark_dist.k",
        public_name: "landmark count",
        purpose: "set how many reference rows define the distance features",
    },
    PublicName {
        key: "op.landmark_dist.gamma",
        public_name: "landmark similarity width",
        purpose: "set how quickly landmark similarity decreases with distance",
    },
    PublicName {
        key: "op.landmark_dist.seed",
        public_name: "landmark selection seed",
        purpose: "make the randomized landmark selection repeatable",
    },
    PublicName {
        key: "op.oblique_hinge.n",
        public_name: "projected hinge count",
        purpose: "set how many learned directions receive piecewise thresholds",
    },
    PublicName {
        key: "op.oblique_hinge.seed",
        public_name: "projected hinge seed",
        purpose: "make the randomized directions repeatable",
    },
    PublicName {
        key: "op.pair_prod.topk",
        public_name: "feature product limit",
        purpose: "limit how many strongly related feature pairs are combined",
    },
    PublicName {
        key: "op.target_bin_encode.bins",
        public_name: "target encoding bin count",
        purpose: "set how many feature groups receive target-based values",
    },
    PublicName {
        key: "op.target_bin_encode.prior",
        public_name: "target encoding prior strength",
        purpose: "set how strongly each group is smoothed toward the overall target mean",
    },
    PublicName {
        key: "op.step.t",
        public_name: "step threshold",
        purpose: "set the cutoff above which a value becomes one",
    },
    PublicName {
        key: "op.lag.k",
        public_name: "lag distance",
        purpose: "set how many row positions back a lagged value is read",
    },
    PublicName {
        key: "op.roll.k",
        public_name: "rotation distance",
        purpose: "set how many positions each row is rotated",
    },
    PublicName {
        key: "op.fft_mag.k",
        public_name: "frequency limit",
        purpose: "set the highest frequency included in each row summary",
    },
    PublicName {
        key: "op.svd_project.k",
        public_name: "component count",
        purpose: "set how many projected directions are retained",
    },
    PublicName {
        key: "op.svd_project.iters",
        public_name: "projection iteration count",
        purpose: "set how many refinement steps estimate the components",
    },
    PublicName {
        key: "op.svd_project.seed",
        public_name: "component seed",
        purpose: "make the randomized component estimate repeatable",
    },
    PublicName {
        key: "op.nystrom.k",
        public_name: "kernel landmark count",
        purpose: "set how many reference rows approximate the kernel",
    },
    PublicName {
        key: "op.nystrom.gamma",
        public_name: "landmark kernel width",
        purpose: "set how quickly kernel similarity decreases with distance",
    },
    PublicName {
        key: "op.nystrom.seed",
        public_name: "kernel landmark seed",
        purpose: "make the randomized landmark selection repeatable",
    },
    PublicName {
        key: "op.symlog.c",
        public_name: "signed log scale",
        purpose: "set the scale at which signed logarithmic compression changes",
    },
    PublicName {
        key: "op.select_corr.k",
        public_name: "screened feature count",
        purpose: "set how many features pass the correlation screen",
    },
    PublicName {
        key: "op.select_corr.by_var",
        public_name: "feature screening basis",
        purpose: "choose whether features are ranked by target relation or training variation",
    },
    PublicName {
        key: "op.impute.strategy",
        public_name: "replacement method",
        purpose: "choose mean, median, or constant replacement for missing values",
    },
    PublicName {
        key: "op.impute.value",
        public_name: "constant replacement value",
        purpose: "set the replacement used when the constant method is selected",
    },
    PublicName {
        key: "op.impute.add_indicator",
        public_name: "missingness marker setting",
        purpose: "choose whether output columns mark the locations of missing values",
    },
    PublicName {
        key: "op.pair_ratio.topk",
        public_name: "ratio feature limit",
        purpose: "limit how many target-related feature ratios are formed",
    },
    PublicName {
        key: "op.spline.knots",
        public_name: "spline knot count",
        purpose: "set how many training quantiles form the spline basis",
    },
    PublicName {
        key: "op.row_vector_input_v2.width",
        public_name: "row input width",
        purpose: "declare the feature width expected by the row-vector operation",
    },
    PublicName {
        key: "op.scalar_v2.carrier",
        public_name: "formula evaluation scope",
        purpose: "choose whether the formula evaluates once per row or for each input column",
    },
    PublicName {
        key: "op.scalar_v2.width",
        public_name: "formula input width",
        purpose: "declare the feature width accepted by the scalar formula",
    },
    PublicName {
        key: "stage.append",
        public_name: "append transform output",
        purpose: "choose whether a stage adds output columns or replaces its input columns",
    },
];

#[cfg(test)]
fn gbm_parameter_keys(gbm: &crate::spec::GbmPlan) -> &'static [&'static str] {
    let crate::spec::GbmPlan {
        rounds: _,
        depth: _,
        lr: _,
        lambda: _,
        bins: _,
        min_leaf: _,
        max_features: _,
    } = gbm;
    &[
        "rounds",
        "depth",
        "lr",
        "lambda",
        "bins",
        "min_leaf",
        "max_features",
    ]
}

#[cfg(test)]
fn planned_head_parameter_keys(head: &crate::spec::PlannedHead) -> &'static [&'static str] {
    use crate::spec::PlannedHead;

    match head {
        PlannedHead::Ridge { lambda_frac: _ } => &["lambda_frac"],
        PlannedHead::RidgeGcv => &[],
        PlannedHead::ElasticNet {
            alpha: _,
            l1_ratio: _,
            sweeps: _,
        } => &["alpha", "l1_ratio", "sweeps"],
        PlannedHead::Gbm(gbm) => gbm_parameter_keys(gbm),
        PlannedHead::Blend {
            lambda_frac: _,
            gbm,
        } => {
            let _ = gbm_parameter_keys(gbm);
            &[
                "lambda_frac",
                "rounds",
                "depth",
                "lr",
                "lambda",
                "bins",
                "min_leaf",
                "max_features",
            ]
        }
        PlannedHead::HistGbm2 {
            leaves: _,
            lr: _,
            bags: _,
            rounds: _,
            row_subsample: _,
            col_subsample: _,
            min_leaf: _,
            val_pct: _,
            val_shuffle: _,
            huber_delta: _,
            stop_mae: _,
        } => &[
            "leaves",
            "lr",
            "bags",
            "rounds",
            "row_subsample",
            "col_subsample",
            "min_leaf",
            "val_pct",
            "val_shuffle",
            "huber_delta",
            "stop_mae",
        ],
        PlannedHead::Krr { m: _ } => &["m"],
        PlannedHead::Gpr { m: _, steps: _ } => &["m", "steps"],
        PlannedHead::GprKernel {
            m: _,
            steps: _,
            source: _,
            spec: _,
        } => &["m", "steps", "kernel"],
        PlannedHead::Ksmooth { k: _, gamma: _ } => &["k", "gamma"],
        PlannedHead::Stack => &[],
        PlannedHead::Portfolio(plan) => {
            // `members` is a collection of nested operation/head plans, whose
            // fields are covered by those type-wide inventories. `fallback`
            // is a derived member index, not an independently tunable value.
            let crate::spec::PortfolioPlan {
                folds: _,
                repeats_auto: _,
                members: _,
                fallback: _,
            } = plan;
            &["folds", "repeats_auto"]
        }
    }
}

#[cfg(test)]
fn evolve_config_parameter_keys(config: &crate::evolve::EvolveConfig) -> &'static [&'static str] {
    // `seed_genomes` contains complete starter programs, not parameter fields.
    let crate::evolve::EvolveConfig {
        pop: _,
        gens: _,
        folds: _,
        seed: _,
        max_transforms: _,
        time_budget_secs: _,
        seed_genomes: _,
        fitness: _,
        target_score: _,
        rank_robust: _,
        hard_ten_priors: _,
        expansion_vocab: _,
        fault_policy: _,
        selection: _,
        ridge_scoring: _,
    } = config;
    &[
        "pop",
        "gens",
        "folds",
        "seed",
        "max_transforms",
        "time_budget_secs",
        "fitness",
        "target_score",
        "rank_robust",
        "hard_ten_priors",
        "expansion_vocab",
        "fault_policy",
        "selection",
        "ridge_scoring",
    ]
}

#[cfg(test)]
fn planned_target_parameter_keys(target: &crate::spec::PlannedTarget) -> &'static [&'static str] {
    let crate::spec::PlannedTarget { kind: _, q: _ } = target;
    &["q"]
}

/// Returns the catalog identifiers for the configurable fields of each planned
/// operation. Expression/source payloads and fitted state are intentionally
/// excluded: they define operation content or learned results, not tunable
/// settings. Context prefixes distinguish repeated names such as `k`, `gamma`,
/// `seed`, and `bins` across operations and model heads.
#[cfg(test)]
fn planned_op_parameter_keys(op: &crate::spec::PlannedOp) -> Vec<&'static str> {
    use crate::spec::PlannedOp;

    match op {
        PlannedOp::Standardize
        | PlannedOp::RobustScale
        | PlannedOp::Rank
        | PlannedOp::Blom
        | PlannedOp::PolyInteract
        | PlannedOp::RowStats
        | PlannedOp::Sigmoid
        | PlannedOp::NormRow
        | PlannedOp::Polar
        | PlannedOp::RowSpectral
        | PlannedOp::RowDiffStats => Vec::new(),
        PlannedOp::QuantileBins { bins: _ } => vec!["op.quantile_bins.bins"],
        PlannedOp::BinOnehot { bins: _ } => vec!["op.bin_onehot.bins"],
        PlannedOp::Hinge { knots: _ } => vec!["op.hinge.knots"],
        PlannedOp::WinsorClip { q: _ } => vec!["op.winsor_clip.q"],
        PlannedOp::Cheb { degree: _ } => vec!["op.cheb.degree"],
        PlannedOp::RffCos {
            n: _,
            gamma: _,
            seed: _,
        } => vec!["op.rff_cos.n", "op.rff_cos.gamma", "op.rff_cos.seed"],
        PlannedOp::RffSin {
            n: _,
            gamma: _,
            seed: _,
        } => vec!["op.rff_sin.n", "op.rff_sin.gamma", "op.rff_sin.seed"],
        PlannedOp::RandProj { k: _, seed: _ } => vec!["op.rand_proj.k", "op.rand_proj.seed"],
        PlannedOp::LandmarkDist {
            k: _,
            gamma: _,
            seed: _,
        } => vec![
            "op.landmark_dist.k",
            "op.landmark_dist.gamma",
            "op.landmark_dist.seed",
        ],
        PlannedOp::ObliqueHinge { n: _, seed: _ } => {
            vec!["op.oblique_hinge.n", "op.oblique_hinge.seed"]
        }
        PlannedOp::PairProd { topk: _ } => vec!["op.pair_prod.topk"],
        PlannedOp::TargetBinEncode { bins: _, prior: _ } => {
            vec!["op.target_bin_encode.bins", "op.target_bin_encode.prior"]
        }
        PlannedOp::Step { t: _ } => vec!["op.step.t"],
        PlannedOp::Lag { k: _ } => vec!["op.lag.k"],
        PlannedOp::Roll { k: _ } => vec!["op.roll.k"],
        PlannedOp::FftMag { k: _ } => vec!["op.fft_mag.k"],
        PlannedOp::SvdProject {
            k: _,
            iters: _,
            seed: _,
        } => vec![
            "op.svd_project.k",
            "op.svd_project.iters",
            "op.svd_project.seed",
        ],
        PlannedOp::Nystrom {
            k: _,
            gamma: _,
            seed: _,
        } => vec!["op.nystrom.k", "op.nystrom.gamma", "op.nystrom.seed"],
        PlannedOp::Symlog { c: _ } => vec!["op.symlog.c"],
        PlannedOp::SelectCorr { k: _, by_var: _ } => {
            vec!["op.select_corr.k", "op.select_corr.by_var"]
        }
        PlannedOp::Impute {
            strategy: _,
            value: _,
            add_indicator: _,
        } => vec![
            "op.impute.strategy",
            "op.impute.value",
            "op.impute.add_indicator",
        ],
        PlannedOp::PairRatio { topk: _ } => vec!["op.pair_ratio.topk"],
        PlannedOp::Spline { knots: _ } => vec!["op.spline.knots"],
        PlannedOp::MapExpr { source: _, expr: _ }
        | PlannedOp::DeriveExpr { source: _, expr: _ } => Vec::new(),
        PlannedOp::RowVectorV2 { state: _ } => Vec::new(),
        PlannedOp::RowVectorInputV2 { width: _ } => vec!["op.row_vector_input_v2.width"],
        PlannedOp::ScalarV2 {
            program: _,
            carrier: _,
            width: _,
        } => vec!["op.scalar_v2.carrier", "op.scalar_v2.width"],
    }
}

#[cfg(test)]
fn planned_stage_parameter_keys(stage: &crate::spec::PlannedStage) -> Vec<&'static str> {
    let crate::spec::PlannedStage { op, append: _ } = stage;
    let mut keys = planned_op_parameter_keys(op);
    keys.push("stage.append");
    keys
}

#[cfg(test)]
mod tests {
    use super::{
        evolve_config_parameter_keys, planned_head_parameter_keys, planned_op_parameter_keys,
        planned_stage_parameter_keys, planned_target_parameter_keys, PARAMETERS,
    };
    use crate::evolve::EvolveConfig;
    use crate::kernel::scalar_program::{ScalarCarrier, ScalarExpressionProgram, ScalarNode};
    use crate::math_expr::Var;
    use crate::spec::{
        GbmPlan, PlannedHead, PlannedOp, PlannedStage, PlannedTarget, PortfolioPlan, TargetKind,
    };

    const BANNED_WORDS: &[&str] = &[
        "mutation",
        "mutate",
        "crossover",
        "genome",
        "generation",
        "population",
        "island",
        "parent",
        "offspring",
        "lineage",
        "DSL",
        "GP",
    ];

    fn contains_banned_word(text: &str) -> bool {
        text.split(|ch: char| !ch.is_alphanumeric()).any(|word| {
            BANNED_WORDS
                .iter()
                .any(|banned| word.eq_ignore_ascii_case(banned))
        })
    }

    fn assert_public_names(entries: &[super::PublicName]) {
        for entry in entries {
            assert!(
                !entry.public_name.trim().is_empty(),
                "{} has no public name",
                entry.key
            );
            assert!(
                !entry.purpose.trim().is_empty(),
                "{} has no purpose",
                entry.key
            );
            assert!(
                !contains_banned_word(entry.public_name),
                "{} public name contains a banned word: {}",
                entry.key,
                entry.public_name
            );
            assert!(
                !contains_banned_word(entry.purpose),
                "{} purpose contains a banned word: {}",
                entry.key,
                entry.purpose
            );
        }
    }

    fn gbm_plan() -> GbmPlan {
        GbmPlan {
            rounds: 1,
            depth: 1,
            lr: 0.1,
            lambda: 1.0,
            bins: 2,
            min_leaf: 1,
            max_features: 1,
        }
    }

    fn planned_heads() -> Vec<PlannedHead> {
        vec![
            PlannedHead::Ridge { lambda_frac: 1.0 },
            PlannedHead::RidgeGcv,
            PlannedHead::ElasticNet {
                alpha: 0.1,
                l1_ratio: 0.5,
                sweeps: 1,
            },
            PlannedHead::Gbm(gbm_plan()),
            PlannedHead::Blend {
                lambda_frac: 1.0,
                gbm: gbm_plan(),
            },
            PlannedHead::HistGbm2 {
                leaves: 2,
                lr: 0.1,
                bags: 1,
                rounds: 1,
                row_subsample: 1.0,
                col_subsample: 1.0,
                min_leaf: 1,
                val_pct: 0,
                val_shuffle: false,
                huber_delta: 0.0,
                stop_mae: false,
            },
            PlannedHead::Krr { m: 1 },
            PlannedHead::Gpr { m: 1, steps: 1 },
            PlannedHead::GprKernel {
                m: 1,
                steps: 1,
                source: "rbf".to_owned(),
                spec: crate::gpr::parse_kernel("rbf").expect("valid kernel specification"),
            },
            PlannedHead::Ksmooth { k: 1, gamma: 1.0 },
            PlannedHead::Stack,
            PlannedHead::Portfolio(PortfolioPlan {
                folds: 2,
                repeats_auto: false,
                members: Vec::new(),
                fallback: 0,
            }),
        ]
    }

    fn planned_operations_with_parameters() -> Vec<PlannedOp> {
        vec![
            PlannedOp::QuantileBins { bins: 2 },
            PlannedOp::BinOnehot { bins: 2 },
            PlannedOp::Hinge { knots: 3 },
            PlannedOp::WinsorClip { q: 0.05 },
            PlannedOp::Cheb { degree: 3 },
            PlannedOp::RffCos {
                n: 8,
                gamma: 0.5,
                seed: 11,
            },
            PlannedOp::RffSin {
                n: 8,
                gamma: 0.5,
                seed: 12,
            },
            PlannedOp::RandProj { k: 4, seed: 13 },
            PlannedOp::LandmarkDist {
                k: 4,
                gamma: 0.5,
                seed: 14,
            },
            PlannedOp::ObliqueHinge { n: 4, seed: 15 },
            PlannedOp::PairProd { topk: 4 },
            PlannedOp::TargetBinEncode {
                bins: 4,
                prior: 2.0,
            },
            PlannedOp::Step { t: 0.5 },
            PlannedOp::Lag { k: 2 },
            PlannedOp::Roll { k: 2 },
            PlannedOp::FftMag { k: 2 },
            PlannedOp::SvdProject {
                k: 4,
                iters: 5,
                seed: 16,
            },
            PlannedOp::Nystrom {
                k: 4,
                gamma: 0.5,
                seed: 17,
            },
            PlannedOp::Symlog { c: 1.0 },
            PlannedOp::SelectCorr { k: 4, by_var: true },
            PlannedOp::Impute {
                strategy: 2,
                value: 0.0,
                add_indicator: true,
            },
            PlannedOp::PairRatio { topk: 4 },
            PlannedOp::Spline { knots: 4 },
            PlannedOp::RowVectorInputV2 { width: 8 },
            PlannedOp::ScalarV2 {
                program: ScalarExpressionProgram::new(ScalarNode::Var(Var::X))
                    .expect("a single variable is a valid scalar program"),
                carrier: ScalarCarrier::Elementwise,
                width: 8,
            },
        ]
    }

    #[test]
    fn every_planned_head_parameter_has_safe_public_wording() {
        assert_public_names(PARAMETERS);

        for (index, entry) in PARAMETERS.iter().enumerate() {
            assert!(
                !PARAMETERS[..index]
                    .iter()
                    .any(|earlier| earlier.key == entry.key),
                "duplicate public parameter entry for {}",
                entry.key
            );
        }

        let heads = planned_heads();
        assert_eq!(heads.len(), 12);
        let mut covered_keys = Vec::new();
        for head in &heads {
            for key in planned_head_parameter_keys(head) {
                assert!(
                    PARAMETERS.iter().any(|entry| entry.key == *key),
                    "{key} is missing from the public parameter catalog"
                );
                covered_keys.push(*key);
            }
        }
        for key in evolve_config_parameter_keys(&EvolveConfig::default()) {
            assert!(
                PARAMETERS.iter().any(|entry| entry.key == *key),
                "{key} is missing from the public parameter catalog"
            );
            covered_keys.push(*key);
        }
        for key in planned_target_parameter_keys(&PlannedTarget {
            kind: TargetKind::Winsor,
            q: 0.02,
        }) {
            assert!(
                PARAMETERS.iter().any(|entry| entry.key == *key),
                "{key} is missing from the public parameter catalog"
            );
            covered_keys.push(*key);
        }
        for op in planned_operations_with_parameters() {
            let stage = PlannedStage { op, append: false };
            covered_keys.extend(planned_stage_parameter_keys(&stage));
        }
        for entry in PARAMETERS {
            assert!(
                covered_keys.contains(&entry.key),
                "{} is not bound to a planned head, target, or search config field",
                entry.key
            );
        }
    }

    #[test]
    fn every_planned_operation_parameter_has_safe_contextual_wording() {
        assert_public_names(PARAMETERS);

        let operations = planned_operations_with_parameters();
        assert_eq!(operations.len(), 25);
        let mut covered_keys = Vec::new();
        let operation_keys: Vec<_> = operations
            .iter()
            .flat_map(|op| planned_op_parameter_keys(op))
            .collect();
        for op in operations {
            let stage = PlannedStage { op, append: true };
            let op_keys = planned_op_parameter_keys(&stage.op);
            assert!(!op_keys.is_empty());
            let stage_keys = planned_stage_parameter_keys(&stage);
            assert!(stage_keys.contains(&"stage.append"));
            for key in stage_keys {
                assert!(
                    PARAMETERS.iter().any(|entry| entry.key == key),
                    "{key} is missing from the public parameter catalog"
                );
                covered_keys.push(key);
            }
            covered_keys.extend(op_keys);
        }

        for key in covered_keys {
            assert!(
                PARAMETERS.iter().any(|entry| entry.key == key),
                "{key} is missing from the public parameter catalog"
            );
        }
        for entry in PARAMETERS
            .iter()
            .filter(|entry| entry.key.starts_with("op."))
        {
            assert!(
                operation_keys.contains(&entry.key),
                "{} is not bound to a PlannedOp tunable field",
                entry.key
            );
        }
    }

    #[test]
    fn parameter_keys_are_bound_to_exhaustive_planned_head_fields() {
        let _key_fn: fn(&PlannedHead) -> &'static [&'static str] = planned_head_parameter_keys;
        let _gbm_key_fn: fn(&GbmPlan) -> &'static [&'static str] = super::gbm_parameter_keys;
        let _config_key_fn: fn(&EvolveConfig) -> &'static [&'static str] =
            evolve_config_parameter_keys;
        let _target_key_fn: fn(&PlannedTarget) -> &'static [&'static str] =
            planned_target_parameter_keys;
        let _op_key_fn: fn(&PlannedOp) -> Vec<&'static str> = planned_op_parameter_keys;
        let _stage_key_fn: fn(&PlannedStage) -> Vec<&'static str> = planned_stage_parameter_keys;
        assert!(planned_head_parameter_keys(&PlannedHead::Stack).is_empty());
    }
}
