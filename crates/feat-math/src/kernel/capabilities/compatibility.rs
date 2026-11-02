/// Compatibility alias for the domain-specific `CapabilityTier` type.
pub type Tier = CapabilityTier;

/// Compatibility alias for the domain-specific `CapabilityClass` type.
pub type Class = CapabilityClass;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn expression_tables_cover_every_lookup() {
        assert_eq!(
            (FUNCTIONS.len(), VARIABLES.len(), BINOPS.len(), UNOPS.len()),
            (19, 10, 5, 1)
        );
        assert_eq!((FITTED_OP_ROWS.len(), FITTED_HEAD_ROWS.len()), (36, 13));
        for (_, name) in FITTED_OP_ROWS.iter().chain(FITTED_HEAD_ROWS) {
            assert!(op_capability(name).is_some(), "{name}");
        }
        for (sample, _, cap) in FUNCTIONS {
            assert_eq!(func_capability(*sample), *cap);
        }
        for (sample, _, cap) in VARIABLES {
            assert_eq!(var_capability(*sample), *cap);
        }
        for (sample, _, cap) in BINOPS {
            assert_eq!(binop_capability(*sample), *cap);
        }
        for (sample, _, cap) in UNOPS {
            assert_eq!(unop_capability(*sample), *cap);
        }
        assert_eq!(var_capability(Var::Col(7)).class, BASE);
        assert!(matches!(
            var_capability(Var::Col(8)).class,
            Class::RuntimeOnlyIntentional { .. }
        ));
    }

    #[test]
    fn every_fitted_op_has_a_declared_row() {
        let parsed = crate::math_expr::MathExpr::parse("c0").unwrap();
        let representatives = vec![
            FittedOp::Standardize {
                means: vec![],
                scales: vec![],
                append: false,
            },
            FittedOp::RobustScale {
                medians: vec![],
                scales: vec![],
                append: false,
            },
            FittedOp::Rank {
                sorted_cols: vec![],
                append: false,
            },
            FittedOp::Blom {
                sorted_cols: vec![],
                append: false,
            },
            FittedOp::QuantileBins {
                edges: vec![],
                bins: 4,
                append: false,
            },
            FittedOp::BinOnehot {
                edges: vec![],
                bins: 4,
                append: false,
            },
            FittedOp::Hinge {
                knots: vec![],
                append: false,
            },
            FittedOp::WinsorClip {
                lo: vec![],
                hi: vec![],
                append: false,
            },
            FittedOp::Cheb {
                degree: 2,
                append: false,
            },
            FittedOp::PolyInteract {
                pairs: vec![],
                append: false,
            },
            FittedOp::RffCos {
                w: vec![],
                b: vec![],
                append: false,
            },
            FittedOp::RffSin {
                w: vec![],
                b: vec![],
                append: false,
            },
            FittedOp::RandProj {
                w: vec![],
                append: false,
            },
            FittedOp::LandmarkDist {
                landmarks: vec![],
                gamma: 1.0,
                append: false,
            },
            FittedOp::ObliqueHinge {
                w: vec![],
                t: vec![],
                append: false,
            },
            FittedOp::RowStats { append: false },
            FittedOp::PairProd {
                pairs: vec![],
                append: false,
            },
            FittedOp::TargetBinEncode {
                edges: vec![],
                enc: vec![],
                append: false,
            },
            FittedOp::Sigmoid { append: false },
            FittedOp::Step {
                t: 0.5,
                append: false,
            },
            FittedOp::NormRow { append: false },
            FittedOp::Polar { append: false },
            FittedOp::Lag {
                k: 1,
                append: false,
            },
            FittedOp::Roll {
                k: 1,
                append: false,
            },
            FittedOp::FftMag {
                k: 1,
                append: false,
            },
            FittedOp::RowSpectral { append: false },
            FittedOp::RowDiffStats { append: false },
            FittedOp::SvdProject {
                means: vec![],
                components: vec![],
                append: false,
            },
            FittedOp::Nystrom {
                landmarks: vec![],
                gamma: 1.0,
                append: false,
            },
            FittedOp::MapExpr {
                source: "c0".into(),
                expr: parsed.clone(),
                append: false,
            },
            FittedOp::DeriveExpr {
                source: "c0".into(),
                expr: parsed,
                append: false,
            },
            FittedOp::Symlog {
                c: 1.0,
                append: false,
            },
            FittedOp::SelectCorr {
                indices: vec![],
                append: false,
            },
            FittedOp::Impute {
                fill: vec![],
                indicator_cols: vec![],
                append: false,
            },
            FittedOp::PairRatio {
                pairs: vec![],
                eps: vec![],
                append: false,
            },
            FittedOp::Spline {
                knots: vec![],
                append: false,
            },
        ];
        assert_eq!(representatives.len(), FITTED_OP_ROWS.len());
        for (representative, (_, name)) in representatives.iter().zip(FITTED_OP_ROWS) {
            assert_eq!(fitted_op_capability(representative).name, *name);
        }
    }

    #[test]
    fn every_cheap_fitted_head_value_maps_to_its_row() {
        use crate::dsl::{ParamValue, Stage};
        let x: crate::Rows = (0..12)
            .map(|i| vec![i as f64 / 10.0, (i * i) as f64 / 100.0])
            .collect();
        let y: Vec<f64> = (0..12)
            .map(|i| if i % 3 == 0 { 0.0 } else { i as f64 / 8.0 })
            .collect();
        let cases = [
            ("ridge", "Ridge", "ridge"),
            ("ridge_gcv", "RidgeGcv", "ridge_gcv"),
            ("elasticnet", "ElasticNet", "elasticnet"),
            ("histgbm", "Gbm", "histgbm"),
            ("ensemble", "Ensemble", "ensemble"),
            ("histgbm2", "HistGbm2", "histgbm2"),
            ("krr", "Krr", "krr"),
            ("gpr", "Gpr", "gpr"),
            ("gpr", "GprKernel", "gpr"),
            ("ksmooth", "Ksmooth", "ksmooth"),
            ("hurdle2", "Hurdle2", "hurdle2"),
            ("stack", "Stack", "stack"),
        ];
        assert_eq!(cases.len(), FITTED_HEAD_ROWS.len() - 1);
        for (op, expected_variant, expected_row) in cases {
            let mut stage = Stage::new(op);
            if matches!(op, "histgbm" | "histgbm2") {
                stage.params.insert("rounds".into(), ParamValue::Int(1));
            }
            if op == "gpr" {
                stage.params.insert("steps".into(), ParamValue::Int(1));
            }
            if expected_variant == "GprKernel" {
                stage
                    .params
                    .insert("kernel".into(), ParamValue::Str("linear".into()));
            }
            let head = crate::heads::fit_head(&stage, &x, &y).unwrap();
            let variant = match &head {
                FittedHead::Ridge(_) => "Ridge",
                FittedHead::RidgeGcv(_) => "RidgeGcv",
                FittedHead::ElasticNet(_) => "ElasticNet",
                FittedHead::Gbm(_) => "Gbm",
                FittedHead::Ensemble { .. } => "Ensemble",
                FittedHead::HistGbm2(_) => "HistGbm2",
                FittedHead::Krr(_) => "Krr",
                FittedHead::Gpr(_) => "Gpr",
                FittedHead::GprKernel(_) => "GprKernel",
                FittedHead::Ksmooth(_) => "Ksmooth",
                FittedHead::Hurdle2(_) => "Hurdle2",
                FittedHead::Stack(_) => "Stack",
                FittedHead::Portfolio(_) => "Portfolio",
            };
            assert_eq!(variant, expected_variant, "{op}");
            assert_eq!(fitted_head_capability(&head).name, expected_row, "{op}");
        }
        // The compact portfolio fit is expensive in debug builds. Its exhaustive
        // match and table entry are checked by expression_tables_cover_every_lookup.
        assert_eq!(FITTED_HEAD_ROWS[12], ("Portfolio", "portfolio"));
    }

    #[test]
    fn rendered_text_is_lint_clean_and_document_is_fresh() {
        let outputs = [
            manifest_json(),
            render_markdown(),
            crate::kernel::profile::render_profiles(),
        ];
        for output in outputs {
            assert!(
                crate::transpile::lint_shipped_text(&output).is_empty(),
                "{:?}",
                crate::transpile::lint_shipped_text(&output)
            );
        }
        assert_eq!(
            render_markdown(),
            include_str!("../../../../docs/internal/kernel2/capabilities.generated.md")
        );
    }

    #[test]
    fn planned_ops_have_probes() {
        use crate::full_rust::{self, OperationKind, ParamDomain};
        use crate::spec::{self, Plan, PlannedHead, PlannedOp};
        let op_name = |op: &PlannedOp| match op {
            // B05–B07: typed scalar recipe arm, outside v1 capability rows.
            PlannedOp::ScalarV2 { .. } => "scalar_expression_v2",
            // B12: these plans are absent from the frozen legacy inventory.
            PlannedOp::RowVectorV2 { .. } | PlannedOp::RowVectorInputV2 { .. } => "feature_row_v2",
            PlannedOp::Standardize => "standardize",
            PlannedOp::RobustScale => "robust_scale",
            PlannedOp::Rank => "rank",
            PlannedOp::Blom => "blom",
            PlannedOp::QuantileBins { .. } => "quantile_bins",
            PlannedOp::BinOnehot { .. } => "bin_onehot",
            PlannedOp::Hinge { .. } => "hinge",
            PlannedOp::WinsorClip { .. } => "winsor_clip",
            PlannedOp::Cheb { .. } => "cheb",
            PlannedOp::PolyInteract => "poly_interact",
            PlannedOp::RffCos { .. } => "rff_cos",
            PlannedOp::RffSin { .. } => "rff_sin",
            PlannedOp::RandProj { .. } => "rand_proj",
            PlannedOp::LandmarkDist { .. } => "landmark_dist",
            PlannedOp::ObliqueHinge { .. } => "oblique_hinge",
            PlannedOp::RowStats => "row_stats",
            PlannedOp::PairProd { .. } => "pair_prod",
            PlannedOp::TargetBinEncode { .. } => "target_bin_encode",
            PlannedOp::Sigmoid => "sigmoid",
            PlannedOp::Step { .. } => "step",
            PlannedOp::NormRow => "norm_row",
            PlannedOp::Polar => "polar",
            PlannedOp::Lag { .. } => "lag",
            PlannedOp::Roll { .. } => "roll",
            PlannedOp::FftMag { .. } => "fft_mag",
            PlannedOp::RowSpectral => "row_spectral",
            PlannedOp::RowDiffStats => "row_diff_stats",
            PlannedOp::SvdProject { .. } => "svd_project",
            PlannedOp::Nystrom { .. } => "nystrom",
            PlannedOp::MapExpr { .. } => "map",
            PlannedOp::DeriveExpr { .. } => "derive",
            PlannedOp::Symlog { .. } => "symlog",
            PlannedOp::SelectCorr { .. } => "select_corr",
            PlannedOp::Impute { .. } => "impute",
            PlannedOp::PairRatio { .. } => "pair_ratio",
            PlannedOp::Spline { .. } => "spline",
        };
        let head_name = |head: &PlannedHead| match head {
            PlannedHead::Ridge { .. } => "ridge",
            PlannedHead::RidgeGcv => "ridge_gcv",
            PlannedHead::ElasticNet { .. } => "elasticnet",
            PlannedHead::Gbm(_) => "histgbm",
            PlannedHead::Blend { .. } => "ensemble",
            PlannedHead::HistGbm2 { .. } => "histgbm2",
            PlannedHead::Krr { .. } => "krr",
            PlannedHead::Gpr { .. } => "gpr",
            PlannedHead::GprKernel { .. } => "gpr",
            PlannedHead::Ksmooth { .. } => "ksmooth",
            PlannedHead::Stack => "stack",
            PlannedHead::Portfolio(_) => "portfolio",
        };
        let mut op_kinds = BTreeSet::new();
        let mut head_kinds = BTreeSet::new();
        for schema in full_rust::operation_schemas() {
            let expression = schema
                .params
                .iter()
                .any(|param| matches!(param.domain, ParamDomain::Expression));
            let source = match schema.kind {
                OperationKind::Head => format!("(pipe X ({}))", schema.op),
                OperationKind::Target => format!("(pipe X ({} kind=log1p) (ridge))", schema.op),
                OperationKind::Transform if expression => {
                    format!("(pipe X ({} expr=\"c0\") (ridge))", schema.op)
                }
                OperationKind::Transform => format!("(pipe X ({} ) (ridge))", schema.op),
            };
            let program = crate::GenomeIr::parse(&source).unwrap();
            let plan = spec::build_plan(&program)
                .unwrap_or_else(|reasons| panic!("{}: {reasons:?}", schema.op));
            if let Plan::Pipeline { stages, head, .. } = plan {
                for stage in stages {
                    op_kinds.insert(op_name(&stage.op));
                }
                head_kinds.insert(head_name(&head));
            }
        }
        let special = crate::GenomeIr::parse("(pipe X (gpr kernel=linear))").unwrap();
        let Plan::Pipeline { head, .. } = spec::build_plan(&special).unwrap() else {
            panic!()
        };
        head_kinds.insert(head_name(&head));
        assert_eq!(op_kinds.len(), 36);
        assert_eq!(head_kinds.len(), 11);
        for name in op_kinds.into_iter().chain(head_kinds) {
            assert!(op_capability(name).is_some(), "{name}");
        }
    }
}
