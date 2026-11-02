mod tests {
    use super::keys::{
        evolve_config_parameter_keys, planned_head_parameter_keys, planned_op_parameter_keys,
        planned_stage_parameter_keys, planned_target_parameter_keys,
    };
    use super::PublicName;
    use super::PARAMETERS;
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

    fn assert_public_names(entries: &[PublicName]) {
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
        let _gbm_key_fn: fn(&GbmPlan) -> &'static [&'static str] = super::keys::gbm_parameter_keys;
        let _config_key_fn: fn(&EvolveConfig) -> &'static [&'static str] =
            evolve_config_parameter_keys;
        let _target_key_fn: fn(&PlannedTarget) -> &'static [&'static str] =
            planned_target_parameter_keys;
        let _op_key_fn: fn(&PlannedOp) -> Vec<&'static str> = planned_op_parameter_keys;
        let _stage_key_fn: fn(&PlannedStage) -> Vec<&'static str> = planned_stage_parameter_keys;
        assert!(planned_head_parameter_keys(&PlannedHead::Stack).is_empty());
    }
}
