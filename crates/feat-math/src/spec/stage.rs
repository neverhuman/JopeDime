use super::*;

/// Plan the leading target pseudo-stage; defaults and clamps mirror
/// `target_transform.rs::fit_target` so the emitted literals equal what the
/// runtime would compute from the same stage.
pub(crate) fn plan_target(stage: &Stage, problems: &mut Vec<String>) -> Option<PlannedTarget> {
    let kind = match stage.params.get("kind") {
        Some(ParamValue::Str(s)) => match s.as_str() {
            "rank_gauss" => TargetKind::RankGauss,
            "log1p" => TargetKind::Log1p,
            "winsor" => TargetKind::Winsor,
            "zscore" => TargetKind::ZScore,
            other => {
                problems.push(format!("target: unknown transform kind '{other}'"));
                return None;
            }
        },
        _ => {
            problems.push("target: requires kind=rank_gauss|log1p|winsor|zscore".to_string());
            return None;
        }
    };
    let q = finite_f64(stage, "q", 0.05, problems).clamp(0.0, 0.25);
    Some(PlannedTarget { kind, q })
}

pub(crate) fn plan_op(stage: &Stage, problems: &mut Vec<String>) -> Option<PlannedOp> {
    let op = match stage.op.as_str() {
        "standardize" | "norm_features" => PlannedOp::Standardize,
        "robust_scale" => PlannedOp::RobustScale,
        "rank" => PlannedOp::Rank,
        "blom" => PlannedOp::Blom,
        "quantile_bins" => PlannedOp::QuantileBins {
            bins: stage.get_usize("bins", 8).clamp(2, 64),
        },
        "bin_onehot" => PlannedOp::BinOnehot {
            bins: stage.get_usize("bins", 6).clamp(2, 32),
        },
        "hinge" => PlannedOp::Hinge {
            knots: stage.get_usize("knots", 4).clamp(1, 16),
        },
        "winsor_clip" | "winsor_moments" => PlannedOp::WinsorClip {
            q: finite_f64(stage, "q", 0.05, problems).clamp(0.0, 0.25),
        },
        "cheb" => PlannedOp::Cheb {
            degree: stage.get_usize("degree", 4).clamp(1, 8),
        },
        "poly_interact" => PlannedOp::PolyInteract,
        "rff_cos" => PlannedOp::RffCos {
            n: stage.get_usize("n", 32).clamp(1, 256),
            gamma: finite_f64(stage, "gamma", 1.0, problems).max(1e-6),
            seed: stage.get_u64("seed", 7),
        },
        "rff_sin" => PlannedOp::RffSin {
            n: stage.get_usize("n", 32).clamp(1, 256),
            gamma: finite_f64(stage, "gamma", 1.0, problems).max(1e-6),
            seed: stage.get_u64("seed", 7),
        },
        "rand_proj" => PlannedOp::RandProj {
            k: stage.get_usize("k", 16).clamp(1, 256),
            seed: stage.get_u64("seed", 7),
        },
        "landmark_dist" => PlannedOp::LandmarkDist {
            k: stage.get_usize("k", 16).clamp(1, 128),
            gamma: finite_f64(stage, "gamma", 1.0, problems).max(1e-6),
            seed: stage.get_u64("seed", 7),
        },
        "oblique_hinge" => PlannedOp::ObliqueHinge {
            n: stage.get_usize("n", 32).clamp(1, 256),
            seed: stage.get_u64("seed", 7),
        },
        "row_stats" => PlannedOp::RowStats,
        "pair_prod" => PlannedOp::PairProd {
            topk: stage.get_usize("topk", 8).clamp(2, 24),
        },
        "target_bin_encode" => PlannedOp::TargetBinEncode {
            bins: stage.get_usize("bins", 8).clamp(2, 32),
            prior: finite_f64(stage, "prior", 10.0, problems).max(0.0),
        },
        "sigmoid" => PlannedOp::Sigmoid,
        "symlog" => PlannedOp::Symlog {
            c: finite_f64(stage, "c", 1.0, problems).clamp(1e-9, 1e9),
        },
        "select_corr" => {
            let by_var = match stage.params.get("by") {
                Some(ParamValue::Str(s)) if s == "var" => true,
                Some(ParamValue::Str(s)) if s == "corr" => false,
                None => false,
                Some(_) => {
                    problems.push("select_corr: by must be corr|var".to_string());
                    false
                }
            };
            PlannedOp::SelectCorr {
                k: stage.get_usize("k", 32).clamp(1, 512),
                by_var,
            }
        }
        "impute" => {
            let strategy = match stage.params.get("strategy") {
                Some(ParamValue::Str(s)) if s == "mean" => 0,
                Some(ParamValue::Str(s)) if s == "median" => 1,
                Some(ParamValue::Str(s)) if s == "const" => 2,
                None => 1,
                Some(_) => {
                    problems.push("impute: strategy must be mean|median|const".to_string());
                    1
                }
            };
            PlannedOp::Impute {
                strategy,
                value: finite_f64(stage, "value", 0.0, problems).clamp(-1e12, 1e12),
                add_indicator: matches!(
                    stage.params.get("add_indicator"),
                    Some(ParamValue::Bool(true))
                ) || stage.get_usize("add_indicator", 0) == 1,
            }
        }
        "pair_ratio" => PlannedOp::PairRatio {
            topk: stage.get_usize("topk", 8).clamp(1, 64),
        },
        "spline" => PlannedOp::Spline {
            knots: stage.get_usize("knots", 5).clamp(3, 12),
        },
        "step" => PlannedOp::Step {
            t: stage.get_f64("t", 0.5),
        },
        "norm_row" => PlannedOp::NormRow,
        "polar" => PlannedOp::Polar,
        "lag" => PlannedOp::Lag {
            k: stage.get_usize("k", 1).max(1),
        },
        "roll" => PlannedOp::Roll {
            k: stage.get_usize("k", 1).max(1),
        },
        "fft_mag" => PlannedOp::FftMag {
            k: stage.get_usize("k", 8),
        },
        "row_spectral" => PlannedOp::RowSpectral,
        "row_diff_stats" => PlannedOp::RowDiffStats,
        "svd_project" => PlannedOp::SvdProject {
            k: stage.get_usize("k", 8).clamp(1, 64),
            iters: stage.get_usize("iters", 20).clamp(1, 200),
            seed: stage.get_u64("seed", 7),
        },
        "nystrom" => PlannedOp::Nystrom {
            k: stage.get_usize("k", 16).clamp(1, 128),
            gamma: finite_f64(stage, "gamma", 1.0, problems).max(1e-6),
            seed: stage.get_u64("seed", 7),
        },
        "map" | "derive" => {
            let source = match stage.params.get("expr") {
                Some(ParamValue::Str(s)) => s.clone(),
                _ => {
                    problems.push(format!(
                        "{}: requires a string parameter expr=\"...\"",
                        stage.op
                    ));
                    return Some(PlannedOp::Sigmoid); // placeholder, plan is discarded
                }
            };
            match MathAst::parse(&source) {
                Ok(expr) => {
                    if stage.op == "map" {
                        PlannedOp::MapExpr { source, expr }
                    } else {
                        PlannedOp::DeriveExpr { source, expr }
                    }
                }
                Err(e) => {
                    problems.push(format!("{}: {e}", stage.op));
                    return Some(PlannedOp::Sigmoid); // placeholder, plan is discarded
                }
            }
        }
        _ => return None,
    };
    Some(op)
}

/// Plan the compact portfolio head: validate the stage parameters, expand the
/// registry roster (members.rs is the single source of truth), and plan every
/// member's inner pipeline with the same op/head planner the outer pipeline
/// uses — literals in the emitted member components therefore equal what the
/// runtime computes from the same stages.
pub(crate) fn plan_portfolio_head(stage: &Stage, problems: &mut Vec<String>) -> PlannedHead {
    let empty = PortfolioPlan {
        folds: 5,
        repeats_auto: true,
        members: Vec::new(),
        fallback: 0,
    };
    let roster = match crate::members::resolve_compact(stage) {
        Ok(r) => r,
        Err(e) => {
            problems.push(format!("{e}"));
            return PlannedHead::Portfolio(empty);
        }
    };
    let repeats_auto = crate::members::compact_repeats_auto(stage);
    let mut members: Vec<PlannedMember> = Vec::new();
    for m in &roster {
        members.push(plan_member(m, false, problems));
    }
    // The small-sample fallback component: reuse ridge_gcv_raw when present,
    // else append it hidden (spec §4 step 6 applies regardless of drops).
    let fallback = match members.iter().position(|m| m.name == "ridge_gcv_raw") {
        Some(i) => i,
        None => {
            let fb = crate::members::small_n_fallback();
            members.push(plan_member(&fb, true, problems));
            members.len() - 1
        }
    };
    PlannedHead::Portfolio(PortfolioPlan {
        folds: 5,
        repeats_auto,
        members,
        fallback,
    })
}

pub(crate) fn plan_member(
    m: &crate::members::ResolvedMember,
    hidden: bool,
    problems: &mut Vec<String>,
) -> PlannedMember {
    let kind = match &m.pipe {
        None => PlannedMemberKind::GroupMean,
        Some(dsl) => {
            let (transforms, head_stage) = match dsl.stages.last() {
                Some(last) if HEAD_OPS.contains(&last.op.as_str()) => {
                    (&dsl.stages[..dsl.stages.len() - 1], Some(last.clone()))
                }
                _ => (&dsl.stages[..], None),
            };
            let mut stages = Vec::with_capacity(transforms.len());
            for stage in transforms {
                if crate::target_transform::is_target_op(&stage.op) {
                    problems.push(format!(
                        "portfolio member '{}': target stages are not emittable inside members",
                        m.name
                    ));
                    continue;
                }
                match plan_op(stage, problems) {
                    Some(op) => {
                        if matches!(op, PlannedOp::MapExpr { .. } | PlannedOp::DeriveExpr { .. }) {
                            problems.push(format!(
                                "portfolio member '{}': formula stages are not emittable inside members",
                                m.name
                            ));
                            continue;
                        }
                        stages.push(PlannedStage {
                            op,
                            append: get_append(stage),
                        });
                    }
                    None => problems.push(stage.op.clone()),
                }
            }
            let head = match &head_stage {
                None => PlannedHead::Ridge { lambda_frac: 0.02 },
                Some(hs) if hs.op == "portfolio" => {
                    problems.push("portfolio members must not nest portfolios".to_string());
                    PlannedHead::Ridge { lambda_frac: 0.02 }
                }
                Some(hs) => {
                    // Re-plan through build_plan's head table by planning a
                    // one-stage pipeline; keeps the two paths identical.
                    match plan_head_stage(hs, problems) {
                        Some(h) => h,
                        None => PlannedHead::Ridge { lambda_frac: 0.02 },
                    }
                }
            };
            PlannedMemberKind::Pipeline {
                stages,
                head: Box::new(head),
            }
        }
    };
    PlannedMember {
        name: m.name.clone(),
        gate: m.gate,
        proj_view: m.view == crate::members::View::Proj,
        hidden,
        kind,
    }
}

/// Plan a single head stage (shared by the outer pipeline and portfolio
/// members). Returns None for unknown head ops.
pub(crate) fn plan_head_stage(stage: &Stage, problems: &mut Vec<String>) -> Option<PlannedHead> {
    let head = match stage.op.as_str() {
        "ridge" | "logistic" => PlannedHead::Ridge {
            lambda_frac: finite_f64(stage, "lambda_frac", 0.02, problems),
        },
        "ridge_gcv" => PlannedHead::RidgeGcv,
        "elasticnet" => PlannedHead::ElasticNet {
            alpha: finite_f64(stage, "alpha", 0.001, problems),
            l1_ratio: finite_f64(stage, "l1_ratio", 0.5, problems),
            sweeps: stage.get_usize("sweeps", 60),
        },
        "histgbm" | "gbm" => PlannedHead::Gbm(gbm_plan(stage, problems)),
        "ensemble" | "ensemble_ridge_gbm" => PlannedHead::Blend {
            lambda_frac: finite_f64(stage, "lambda_frac", 0.02, problems),
            gbm: gbm_plan(stage, problems),
        },
        "histgbm2" => {
            let huber = matches!(
                stage.params.get("loss"),
                Some(ParamValue::Str(s)) if s == "huber"
            );
            PlannedHead::HistGbm2 {
                leaves: stage
                    .get_usize("leaves", crate::boost2::DEFAULT_LEAVES)
                    .clamp(2, 255),
                lr: finite_f64(stage, "lr", crate::boost2::DEFAULT_LR, problems),
                bags: stage
                    .get_usize("bags", crate::boost2::DEFAULT_BAGS)
                    .clamp(1, 10),
                rounds: stage
                    .get_usize("rounds", crate::boost2::MAX_ROUNDS)
                    .clamp(1, crate::boost2::ROUNDS_CAP),
                row_subsample: finite_f64(
                    stage,
                    "row_subsample",
                    crate::boost2::SUBSAMPLE,
                    problems,
                )
                .clamp(0.5, 1.0),
                col_subsample: finite_f64(
                    stage,
                    "col_subsample",
                    crate::boost2::DEFAULT_COL_SUBSAMPLE,
                    problems,
                )
                .clamp(0.5, 1.0),
                min_leaf: stage
                    .get_usize("min_leaf", crate::boost2::MIN_LEAF)
                    .clamp(1, 20),
                val_pct: stage
                    .get_usize("val_pct", crate::boost2::DEFAULT_VAL_PCT)
                    .clamp(5, 40),
                val_shuffle: matches!(
                    stage.params.get("val_shuffle"),
                    Some(ParamValue::Bool(true))
                ),
                huber_delta: if huber {
                    finite_f64(stage, "delta", crate::boost2::DEFAULT_HUBER_DELTA, problems)
                        .clamp(1e-6, 1e6)
                } else {
                    0.0
                },
                stop_mae: matches!(
                    stage.params.get("stop"),
                    Some(ParamValue::Str(s)) if s == "mae"
                ),
            }
        }
        "krr" => PlannedHead::Krr {
            m: stage.get_usize("m", crate::krr::DEFAULT_M).clamp(50, 2000),
        },
        "gpr" => {
            let m = stage.get_usize("m", crate::gpr::DEFAULT_M).clamp(50, 2000);
            let steps = stage
                .get_usize("steps", crate::gpr::DEFAULT_STEPS)
                .clamp(1, 200);
            let kernel_src = match stage.params.get("kernel") {
                Some(ParamValue::Str(s)) => s.clone(),
                _ => "rbf".to_string(),
            };
            if kernel_src == "rbf" {
                PlannedHead::Gpr { m, steps }
            } else {
                match crate::gpr::parse_kernel(&kernel_src) {
                    Ok(spec) => PlannedHead::GprKernel {
                        m,
                        steps,
                        source: kernel_src,
                        spec,
                    },
                    Err(e) => {
                        problems.push(format!("gpr kernel: {e}"));
                        PlannedHead::Gpr { m, steps }
                    }
                }
            }
        }
        "ksmooth" => PlannedHead::Ksmooth {
            k: stage
                .get_usize("k", crate::ksmooth::DEFAULT_K)
                .clamp(1, 512),
            gamma: finite_f64(stage, "gamma", crate::ksmooth::DEFAULT_GAMMA, problems)
                .clamp(1e-6, 1e6),
        },
        "stack" => PlannedHead::Stack,
        _ => return None,
    };
    Some(head)
}
