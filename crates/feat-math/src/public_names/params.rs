//! Plain-language names and purposes for tunable model parameters.

use super::PublicName;

/// Public descriptions for every tunable parameter in a planned model head.
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

#[cfg(test)]
mod tests {
    use super::{
        evolve_config_parameter_keys, planned_head_parameter_keys, planned_target_parameter_keys,
        PARAMETERS,
    };
    use crate::evolve::EvolveConfig;
    use crate::spec::{GbmPlan, PlannedHead, PlannedTarget, PortfolioPlan, TargetKind};

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
        covered_keys.extend(evolve_config_parameter_keys(&EvolveConfig::default()));
        covered_keys.extend(planned_target_parameter_keys(&PlannedTarget {
            kind: TargetKind::Winsor,
            q: 0.02,
        }));
        for entry in PARAMETERS {
            assert!(
                covered_keys.contains(&entry.key),
                "{} is not bound to a planned head or search config field",
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
        assert!(planned_head_parameter_keys(&PlannedHead::Stack).is_empty());
    }
}
