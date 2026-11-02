fn build_op(
    name: &'static str,
    v3: &BTreeMap<String, bool>,
    jain: &BTreeMap<String, bool>,
) -> OpCap {
    let reg = registry::operator(name);
    let kind = if let Some(row) = reg {
        match row.role {
            OperatorRole::Transform => OpKind::Transform,
            OperatorRole::Head => OpKind::Head,
            OperatorRole::Target => OpKind::Target,
            OperatorRole::Directive => OpKind::Directive,
        }
    } else if name == "input" {
        OpKind::Input
    } else if name == "target" {
        OpKind::Target
    } else if crate::heads::is_head_op(name)
        || matches!(
            name,
            "logit_ridge"
                | "multiclass_ridge"
                | "multiclass_diagonal_lda"
                | "multiclass_nearest_centroid"
                | "multiclass_ovr_logistic"
                | "tab_boost"
                | "extra_trees"
                | "gam_boost"
                | "pls"
        )
    {
        OpKind::Head
    } else {
        OpKind::Transform
    };
    let lifecycle = reg.map(|row| {
        if row.lifecycle.quarantined {
            RegistryLifecycle::Quarantined
        } else if row.lifecycle.generatable {
            RegistryLifecycle::Generatable
        } else {
            RegistryLifecycle::Compatibility
        }
    });
    let v3_admission = if V3_RECEIPT.contains(&name) {
        V3Admission::RequiresReceipt
    } else if V3_BASELINE.contains(&name) {
        V3Admission::Baseline
    } else {
        V3Admission::Absent
    };
    let stage_vocab = if LEGACY_STAGES.contains(&name) || LEGACY_HEADS.contains(&name) {
        StageVocab::Legacy
    } else if crate::evolve::EXPANSION_TRANSFORMS.contains(&name) {
        StageVocab::Expansion
    } else {
        StageVocab::None
    };
    let kernel2 = if LEGACY_STAGES.contains(&name)
        || LEGACY_HEADS.contains(&name)
        || matches!(name, "map" | "derive")
    {
        Kernel2Use::Baseline
    } else {
        Kernel2Use::NotGenerated {
            reason: "outside revision 1",
        }
    };
    let drift = if LEGACY_DRIFT.contains(&name) {
        Drift::LegacyDrift {
            reason: "frozen grammar differs from registry",
        }
    } else if GRAMMAR_ONLY.contains(&name) {
        Drift::GrammarOnly {
            reason: "frozen grammar only",
        }
    } else {
        Drift::None
    };
    let width_effect = reg
        .map(|r| WidthEffect::Registry(r.shape))
        .unwrap_or_else(|| v3_width(name));
    let emitters = op_emitters(name, kind, lifecycle);
    OpCap {
        name,
        canonical: reg.map_or(name, |r| r.semantic_key),
        kind,
        wire_id: reg.map(|r| r.id.0),
        registry_lifecycle: lifecycle,
        v3_admission,
        grammar: GrammarFlags {
            v3: v3.get(name).copied(),
            jain_v1: jain.get(name).copied(),
        },
        stage_vocab,
        kernel2,
        drift,
        width_effect,
        requires_labels: reg.map_or(
            matches!(
                name,
                "target"
                    | "input"
                    | "corr_screen"
                    | "pls_project"
                    | "select_corr"
                    | "logit_ridge"
                    | "multiclass_ridge"
                    | "tab_boost"
                    | "extra_trees"
                    | "gam_boost"
                    | "pls"
                    | "portfolio"
            ),
            |r| r.requires_labels,
        ),
        requires_row_order: matches!(name, "lag" | "roll"),
        emitters,
        notes: if name == "portfolio" {
            "compact head; nested form has a separate row"
        } else if name == "target" {
            "runtime metadata differs from recipe support"
        } else {
            ""
        },
    }
}

fn v3_width(name: &str) -> WidthEffect {
    match name {
        "input" => WidthEffect::Input,
        "maxabs_scale" | "power_transform" => WidthEffect::OneToOne,
        "piecewise_linear" => WidthEffect::ExpansionPerFeature { parameter: "bins" },
        "corr_screen" | "select_corr" => WidthEffect::Select { parameter: "k" },
        "pls_project" => WidthEffect::Projection { parameter: "k" },
        "impute" => WidthEffect::Impute,
        _ => WidthEffect::None,
    }
}

fn op_emitters(name: &str, kind: OpKind, lifecycle: Option<RegistryLifecycle>) -> EmitterSupport {
    if lifecycle == Some(RegistryLifecycle::Quarantined) {
        return EmitterSupport {
            rust_recipe: Support::Unsupported {
                reason: "route unavailable",
            },
            python_recipe: Support::Unsupported {
                reason: "route unavailable",
            },
            cpp17_recipe: Support::Unsupported {
                reason: "route unavailable",
            },
            cpp17_fitted: Support::Unsupported {
                reason: "route unavailable",
            },
            javascript: NO_JS,
        };
    }
    if matches!(
        name,
        "maxabs_scale"
            | "power_transform"
            | "piecewise_linear"
            | "corr_screen"
            | "pls_project"
            | "logit_ridge"
            | "multiclass_ridge"
            | "multiclass_diagonal_lda"
            | "multiclass_nearest_centroid"
            | "multiclass_ovr_logistic"
            | "tab_boost"
            | "extra_trees"
            | "gam_boost"
            | "pls"
    ) {
        let fitted = if matches!(
            name,
            "multiclass_ridge"
                | "multiclass_diagonal_lda"
                | "multiclass_nearest_centroid"
                | "multiclass_ovr_logistic"
        ) {
            Support::Unsupported {
                reason: "scalar route",
            }
        } else {
            YES
        };
        return EmitterSupport {
            rust_recipe: Support::Unsupported {
                reason: "route unavailable",
            },
            python_recipe: Support::Unsupported {
                reason: "route unavailable",
            },
            cpp17_recipe: Support::Unsupported {
                reason: "route unavailable",
            },
            cpp17_fitted: fitted,
            javascript: NO_JS,
        };
    }
    let recipe = if name == "input" || name == "hurdle2" {
        Support::Deferred { reason: "planning" }
    } else {
        YES
    };
    let cpp17_recipe = if CPP_TRANSFORMS.contains(&name)
        || matches!(
            name,
            "norm_features" | "winsor_moments" | "ridge" | "logistic" | "elasticnet"
        ) {
        YES
    } else if matches!(name, "map" | "derive") {
        Support::Planned { slice: "X2a" }
    } else if name == "target" {
        Support::Planned { slice: "X2b" }
    } else if crate::evolve::EXPANSION_TRANSFORMS.contains(&name) {
        Support::Planned { slice: "X2c" }
    } else if matches!(
        name,
        "polar" | "lag" | "roll" | "fft_mag" | "row_spectral" | "svd_project" | "nystrom"
    ) {
        Support::Planned { slice: "X2d" }
    } else if name == "ridge_gcv" {
        Support::Planned { slice: "X2e" }
    } else if name == "ksmooth" {
        Support::Planned { slice: "X2f" }
    } else if name == "histgbm" || name == "gbm" {
        Support::Planned { slice: "X2g" }
    } else if name == "histgbm2" {
        Support::Planned { slice: "X2h" }
    } else if name == "gpr" {
        Support::Planned { slice: "X2i" }
    } else {
        Support::Deferred {
            reason: "fitted route only",
        }
    };
    let cpp17_fitted = YES;
    let javascript = if cpp17_recipe == YES {
        JS_SUBSET
    } else {
        NO_JS
    };
    let _ = kind;
    EmitterSupport {
        rust_recipe: recipe,
        python_recipe: recipe,
        cpp17_recipe,
        cpp17_fitted,
        javascript,
    }
}

// The list creates both the exhaustive match and the table walked by coverage tests.
macro_rules! fitted_op_rows {
    ($( $variant:ident => $name:literal, )+) => {
        pub const FITTED_OP_ROWS: &[(&str, &str)] = &[$((stringify!($variant), $name),)+];
        pub fn fitted_op_capability(op: &FittedOp) -> &'static OpCap {
            let name = match op { $(FittedOp::$variant { .. } => $name,)+ };
            op_capability(name).expect("each fitted stage has a row")
        }
    };
}
fitted_op_rows! {
    Standardize => "standardize", RobustScale => "robust_scale", Rank => "rank", Blom => "blom",
    QuantileBins => "quantile_bins", BinOnehot => "bin_onehot", Hinge => "hinge", WinsorClip => "winsor_clip",
    Cheb => "cheb", PolyInteract => "poly_interact", RffCos => "rff_cos", RffSin => "rff_sin",
    RandProj => "rand_proj", LandmarkDist => "landmark_dist", ObliqueHinge => "oblique_hinge",
    RowStats => "row_stats", PairProd => "pair_prod", TargetBinEncode => "target_bin_encode",
    Sigmoid => "sigmoid", Step => "step", NormRow => "norm_row", Polar => "polar",
    Lag => "lag", Roll => "roll", FftMag => "fft_mag", RowSpectral => "row_spectral",
    RowDiffStats => "row_diff_stats", SvdProject => "svd_project", Nystrom => "nystrom",
    MapExpr => "map", DeriveExpr => "derive", Symlog => "symlog", SelectCorr => "select_corr",
    Impute => "impute", PairRatio => "pair_ratio", Spline => "spline",
}

macro_rules! fitted_head_rows {
    ($( $pattern:pat => ($variant:literal, $name:literal), )+) => {
        pub const FITTED_HEAD_ROWS: &[(&str, &str)] = &[$(($variant, $name),)+];
        pub fn fitted_head_capability(head: &FittedHead) -> &'static OpCap {
            let name = match head { $($pattern => $name,)+ };
            op_capability(name).expect("each fitted head has a row")
        }
    };
}
fitted_head_rows! {
    FittedHead::Ridge(_) => ("Ridge", "ridge"),
    FittedHead::RidgeGcv(_) => ("RidgeGcv", "ridge_gcv"),
    FittedHead::ElasticNet(_) => ("ElasticNet", "elasticnet"),
    FittedHead::Gbm(_) => ("Gbm", "histgbm"),
    FittedHead::Ensemble { .. } => ("Ensemble", "ensemble"),
    FittedHead::HistGbm2(_) => ("HistGbm2", "histgbm2"),
    FittedHead::Krr(_) => ("Krr", "krr"),
    FittedHead::Gpr(_) => ("Gpr", "gpr"),
    FittedHead::GprKernel(_) => ("GprKernel", "gpr"),
    FittedHead::Ksmooth(_) => ("Ksmooth", "ksmooth"),
    FittedHead::Hurdle2(_) => ("Hurdle2", "hurdle2"),
    FittedHead::Stack(_) => ("Stack", "stack"),
    FittedHead::Portfolio(_) => ("Portfolio", "portfolio"),
}
