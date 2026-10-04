//! Genome analysis: resolve a `GenomeIr` into a fully-literal `Plan` that both
//! emitters consume, plus the shared naming/doc tables and helper dependency
//! closure. Parameter defaults and clamps are copied verbatim from
//! battle-runtime `ops.rs` / `heads.rs` so the emitted literals equal what the
//! runtime would compute from the same stage.

use crate::dsl::{ParamValue, Stage};
use crate::gpr::{KernelPrim, KernelSpec};
use crate::mathexpr::{ExprFeatures, MathAst, MathFunc, MathVar};
use crate::GenomeIr;
use std::collections::BTreeSet;

// --------------------------------- plan ----------------------------------

#[derive(Clone, Debug)]
pub enum Plan {
    Pipeline {
        /// Optional target-side transform (always the FIRST stage of the
        /// source pipeline; same split semantics as `model.rs::split_target`).
        target: Option<PlannedTarget>,
        stages: Vec<PlannedStage>,
        head: PlannedHead,
    },
    Symbolic {
        source: String,
        expr: MathAst,
        affine: bool,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TargetKind {
    RankGauss,
    Log1p,
    Winsor,
    ZScore,
}

#[derive(Clone, Debug)]
pub struct PlannedTarget {
    pub kind: TargetKind,
    /// Winsor tail fraction, pre-clamped to [0, 0.25] exactly as
    /// `target_transform.rs::fit_target` clamps it; unused by other kinds.
    pub q: f64,
}

#[derive(Clone, Debug)]
pub struct PlannedStage {
    pub op: PlannedOp,
    pub append: bool,
}

#[derive(Clone, Debug)]
pub enum PlannedOp {
    Standardize,
    RobustScale,
    Rank,
    Blom,
    QuantileBins {
        bins: usize,
    },
    BinOnehot {
        bins: usize,
    },
    Hinge {
        knots: usize,
    },
    WinsorClip {
        q: f64,
    },
    Cheb {
        degree: usize,
    },
    PolyInteract,
    RffCos {
        n: usize,
        gamma: f64,
        seed: u64,
    },
    RffSin {
        n: usize,
        gamma: f64,
        seed: u64,
    },
    RandProj {
        k: usize,
        seed: u64,
    },
    LandmarkDist {
        k: usize,
        gamma: f64,
        seed: u64,
    },
    ObliqueHinge {
        n: usize,
        seed: u64,
    },
    RowStats,
    PairProd {
        topk: usize,
    },
    TargetBinEncode {
        bins: usize,
        prior: f64,
    },
    Sigmoid,
    /// Per-column hard indicator `1.0 if v > t else 0.0` at a fixed parameter
    /// threshold `t` (default 0.5). Stateless — the honest replacement for the
    /// `clip((v - t) * 1e15, 0, 1)` idiom.
    Step {
        t: f64,
    },
    NormRow,
    Polar,
    Lag {
        k: usize,
    },
    Roll {
        k: usize,
    },
    /// `k` is the raw request; the artifact clamps to [1, d/2] at fit time
    /// (the training width is unknown at emission).
    FftMag {
        k: usize,
    },
    RowSpectral,
    RowDiffStats,
    /// `k` is pre-clamped to [1, 64]; the artifact applies `.min(cols)` at fit.
    SvdProject {
        k: usize,
        iters: usize,
        seed: u64,
    },
    /// `k` is pre-clamped to [1, 128]; the artifact applies `.min(n)` at fit.
    Nystrom {
        k: usize,
        gamma: f64,
        seed: u64,
    },
    MapExpr {
        source: String,
        expr: MathAst,
    },
    DeriveExpr {
        source: String,
        expr: MathAst,
    },
    /// Signed shifted-log compression asinh(v / c); `c` pre-clamped to
    /// [1e-9, 1e9]. Stateless.
    Symlog {
        c: f64,
    },
    /// Top-k column screening; `k` pre-clamped to [1, 512], the artifact
    /// applies `.min(cols)` at fit. `by_var` false = |pearson(x_j, y)|
    /// (target-aware), true = train variance.
    SelectCorr {
        k: usize,
        by_var: bool,
    },
    /// Zero-as-missing imputation. `strategy`: 0=mean, 1=median, 2=const;
    /// `value` is the const fill (pre-clamped to [-1e12, 1e12], passed but
    /// unused for mean/median); `add_indicator` appends one 0/1 missingness
    /// column per train-zero-containing column.
    Impute {
        strategy: usize,
        value: f64,
        add_indicator: bool,
    },
    /// Target-guided regularized ratios x_a * x_b / (x_b^2 + eps) over the
    /// screened top-32 columns; `topk` pre-clamped to [1, 64].
    PairRatio {
        topk: usize,
    },
    /// Natural cubic spline basis at train quantile knots; `knots` pre-clamped
    /// to [3, 12].
    Spline {
        knots: usize,
    },
    // B12: constructed only by the explicit v2 row-vector planner.
    RowVectorV2 {
        state: crate::kernel::row_vector::FittedStage,
    },
    RowVectorInputV2 {
        width: usize,
    },
}

#[derive(Clone, Debug)]
pub struct GbmPlan {
    pub rounds: usize,
    pub depth: usize,
    pub lr: f64,
    pub lambda: f64,
    pub bins: usize,
    pub min_leaf: usize,
    pub max_features: usize,
}

#[derive(Clone, Debug)]
pub enum PlannedHead {
    Ridge {
        lambda_frac: f64,
    },
    /// `ridge_gcv`: parameter-free ridge whose penalty is chosen by generalized
    /// cross-validation over the fixed grid logspace(-5, 3, 25); fully resolved
    /// by construction.
    RidgeGcv,
    ElasticNet {
        alpha: f64,
        l1_ratio: f64,
        sweeps: usize,
    },
    Gbm(GbmPlan),
    Blend {
        lambda_frac: f64,
        gbm: GbmPlan,
    },
    /// `histgbm2`: leaf-wise Newton boosting with exact splits and bagging.
    /// Defaults replay the historical head bit-for-bit.
    HistGbm2 {
        leaves: usize,
        lr: f64,
        bags: usize,
        rounds: usize,
        row_subsample: f64,
        col_subsample: f64,
        min_leaf: usize,
        val_pct: usize,
        val_shuffle: bool,
        /// Huber gradient clip; 0.0 means plain squared loss.
        huber_delta: f64,
        /// Early-stop the holdout on MAE instead of RMSE.
        stop_mae: bool,
    },
    /// `krr`: Gaussian kernel ridge regression with grid-selected width/penalty.
    Krr {
        m: usize,
    },
    /// `gpr`: Gaussian-process regression with a learned ARD kernel.
    Gpr {
        m: usize,
        steps: usize,
    },
    /// `gpr kernel="..."`: the composite-kernel GP (sum/product of primitives).
    GprKernel {
        m: usize,
        steps: usize,
        source: String,
        spec: KernelSpec,
    },
    /// `ksmooth`: Nadaraya-Watson kernel-weighted nearest-neighbor smoother.
    Ksmooth {
        k: usize,
        gamma: f64,
    },
    /// `stack`: greedy ensemble selection over the fixed five-model library.
    Stack,
    /// `portfolio` (compact head): the pinned member registry expanded to a
    /// fixed component roster, gated at fit time on train stats, measured by
    /// seeded shuffled repeated K-fold OOF predictions, and combined by a
    /// non-negative ridge blend in deviation-from-mean space.
    Portfolio(PortfolioPlan),
}

#[derive(Clone, Debug)]
pub struct PortfolioPlan {
    pub folds: usize,
    pub repeats_auto: bool,
    /// Registry members after preset/add/drop resolution, in registry order.
    pub members: Vec<PlannedMember>,
    /// Index (0-based, into `members`) of the small-sample fallback
    /// component; when `ridge_gcv_raw` was dropped from the roster an extra
    /// hidden member is appended for this role (never gated in).
    pub fallback: usize,
}

#[derive(Clone, Debug)]
pub struct PlannedMember {
    /// Registry name — plan metadata only, NEVER emitted (banned vocabulary).
    pub name: String,
    pub gate: crate::members::Gate,
    pub proj_view: bool,
    /// Hidden members exist only for the small-n fallback: they are emitted
    /// as components but never appended to the gated roster.
    pub hidden: bool,
    pub kind: PlannedMemberKind,
}

#[derive(Clone, Debug)]
pub enum PlannedMemberKind {
    Pipeline {
        stages: Vec<PlannedStage>,
        head: Box<PlannedHead>,
    },
    GroupMean,
}

const HEAD_OPS: [&str; 14] = [
    "ridge",
    "logistic",
    "ridge_gcv",
    "elasticnet",
    "histgbm",
    "gbm",
    "ensemble",
    "ensemble_ridge_gbm",
    "histgbm2",
    "krr",
    "gpr",
    "ksmooth",
    "stack",
    "portfolio",
];

fn get_append(stage: &Stage) -> bool {
    matches!(stage.params.get("append"), Some(ParamValue::Bool(true)))
        || stage.get_usize("append", 0) == 1
}

fn finite_f64(stage: &Stage, key: &str, default: f64, problems: &mut Vec<String>) -> f64 {
    let v = stage.get_f64(key, default);
    if !v.is_finite() {
        problems.push(format!("{}: parameter {key} is not finite", stage.op));
    }
    v
}

fn gbm_plan(stage: &Stage, problems: &mut Vec<String>) -> GbmPlan {
    GbmPlan {
        rounds: stage.get_usize("rounds", 150).clamp(1, 500),
        depth: stage.get_usize("depth", 3).clamp(1, 8),
        lr: finite_f64(stage, "lr", 0.1, problems),
        lambda: finite_f64(stage, "lambda", 1.0, problems),
        bins: stage.get_usize("bins", 64).clamp(2, 255),
        min_leaf: stage.get_usize("min_leaf", 10).max(1),
        max_features: stage.get_usize("max_features", 256).max(1),
    }
}

pub fn build_plan(genome: &GenomeIr) -> Result<Plan, Vec<String>> {
    match genome {
        // Explicit nested portfolios are engine-executable but not yet
        // emittable (G5); the compact head form `(pipe X ... (portfolio
        // preset=core ...))` is the supported emission path.
        GenomeIr::Portfolio(_) => Err(vec![
            "portfolio: explicit nested member emission is not supported yet; \
             use the compact head form (pipe X ... (portfolio preset=core ...))"
                .to_string(),
        ]),
        GenomeIr::Symbolic { source, affine, .. } => match MathAst::parse(source) {
            Ok(expr) => Ok(Plan::Symbolic {
                source: source.clone(),
                expr,
                affine: *affine,
            }),
            Err(e) => Err(vec![format!("symbolic expression: {e}")]),
        },
        GenomeIr::Pipeline(dsl) => {
            let mut problems: Vec<String> = Vec::new();
            // Same split semantics as model.rs::split_target: an optional
            // leading target pseudo-stage; anywhere else it is an error.
            let (target_stage, rest) = match dsl.stages.first() {
                Some(first) if crate::target_transform::is_target_op(&first.op) => {
                    (Some(first), &dsl.stages[1..])
                }
                _ => (None, &dsl.stages[..]),
            };
            if rest
                .iter()
                .any(|s| crate::target_transform::is_target_op(&s.op))
            {
                problems.push("target: transform must be the first stage".to_string());
            }
            let target = target_stage.and_then(|stage| plan_target(stage, &mut problems));
            let (transforms, head_stage) = match rest.last() {
                Some(last) if HEAD_OPS.contains(&last.op.as_str()) => {
                    (&rest[..rest.len() - 1], Some(last.clone()))
                }
                _ => (rest, None),
            };
            let mut stages = Vec::with_capacity(transforms.len());
            for stage in transforms {
                if HEAD_OPS.contains(&stage.op.as_str()) {
                    problems.push(format!(
                        "{}: model heads are only valid as the final stage",
                        stage.op
                    ));
                    continue;
                }
                match plan_op(stage, &mut problems) {
                    Some(op) => stages.push(PlannedStage {
                        op,
                        append: get_append(stage),
                    }),
                    None => problems.push(stage.op.clone()),
                }
            }
            let head = match &head_stage {
                None => PlannedHead::Ridge { lambda_frac: 0.02 },
                Some(stage) => match stage.op.as_str() {
                    "portfolio" => plan_portfolio_head(stage, &mut problems),
                    // Defaults and clamps mirror battle-runtime heads.rs
                    // (shared with portfolio member planning).
                    _ => match plan_head_stage(stage, &mut problems) {
                        Some(h) => h,
                        None => {
                            problems.push(stage.op.clone());
                            PlannedHead::Ridge { lambda_frac: 0.02 }
                        }
                    },
                },
            };
            if problems.is_empty() {
                Ok(Plan::Pipeline {
                    target,
                    stages,
                    head,
                })
            } else {
                Err(problems)
            }
        }
    }
}

/// Plan the leading target pseudo-stage; defaults and clamps mirror
/// `target_transform.rs::fit_target` so the emitted literals equal what the
/// runtime would compute from the same stage.
fn plan_target(stage: &Stage, problems: &mut Vec<String>) -> Option<PlannedTarget> {
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

fn plan_op(stage: &Stage, problems: &mut Vec<String>) -> Option<PlannedOp> {
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
fn plan_portfolio_head(stage: &Stage, problems: &mut Vec<String>) -> PlannedHead {
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

fn plan_member(
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
fn plan_head_stage(stage: &Stage, problems: &mut Vec<String>) -> Option<PlannedHead> {
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

// ------------------------------ naming tables ------------------------------

/// Stable per-kind key: also the base of the emitted function names.
pub fn op_fn_name(op: &PlannedOp) -> &'static str {
    match op {
        // B13: a goertzel_bank stage keys its own helper; B12 keys stay unchanged.
        PlannedOp::RowVectorV2 {
            state: crate::kernel::row_vector::FittedStage::Goertzel(_),
        } => "row_vector_goertzel_v2",
        // B12: version-two FeatureWithinRow stage.
        PlannedOp::RowVectorV2 { .. } | PlannedOp::RowVectorInputV2 { .. } => "row_vector_v2",
        PlannedOp::Standardize => "standard_scaler",
        PlannedOp::RobustScale => "robust_scaler",
        PlannedOp::Rank => "mid_rank_transform",
        PlannedOp::Blom => "blom_normal_scores",
        PlannedOp::QuantileBins { .. } => "quantile_bin_indexer",
        PlannedOp::BinOnehot { .. } => "quantile_onehot_encoder",
        PlannedOp::Hinge { .. } => "hinge_basis",
        PlannedOp::WinsorClip { .. } => "winsor_limiter",
        PlannedOp::Cheb { .. } => "chebyshev_basis",
        PlannedOp::PolyInteract => "pairwise_products",
        PlannedOp::RffCos { .. } => "random_fourier_cosine_features",
        PlannedOp::RffSin { .. } => "random_fourier_sine_features",
        PlannedOp::RandProj { .. } => "gaussian_random_projection",
        PlannedOp::LandmarkDist { .. } => "gaussian_landmark_features",
        PlannedOp::ObliqueHinge { .. } => "oblique_hinge_features",
        PlannedOp::RowStats => "row_summary_statistics",
        PlannedOp::PairProd { .. } => "correlation_screened_products",
        PlannedOp::TargetBinEncode { .. } => "smoothed_target_bin_encoder",
        PlannedOp::Sigmoid => "logistic_squash",
        PlannedOp::Step { .. } => "step_indicator",
        PlannedOp::NormRow => "unit_norm_rows",
        PlannedOp::Polar => "polar_coordinates",
        PlannedOp::Lag { .. } => "shift_columns",
        PlannedOp::Roll { .. } => "rotate_columns",
        PlannedOp::FftMag { .. } => "fourier_magnitudes",
        PlannedOp::RowSpectral => "row_spectral_summary",
        PlannedOp::RowDiffStats => "row_difference_statistics",
        PlannedOp::SvdProject { .. } => "principal_component_projection",
        PlannedOp::Nystrom { .. } => "nystrom_landmark_features",
        PlannedOp::MapExpr { .. } => "elementwise_formula",
        PlannedOp::DeriveExpr { .. } => "derived_column",
        PlannedOp::Symlog { .. } => "signed_log",
        PlannedOp::SelectCorr { .. } => "column_select",
        PlannedOp::Impute { .. } => "zero_imputer",
        PlannedOp::PairRatio { .. } => "ratio_screened_pairs",
        PlannedOp::Spline { .. } => "natural_cubic_spline",
    }
}

/// Rust fitted-state struct name (None for stateless transforms).
pub fn op_struct_name(op: &PlannedOp) -> Option<&'static str> {
    match op {
        // B12: version-two FeatureWithinRow stage.
        PlannedOp::RowVectorV2 { .. } | PlannedOp::RowVectorInputV2 { .. } => None,
        PlannedOp::Standardize => Some("StandardScaler"),
        PlannedOp::RobustScale => Some("RobustScaler"),
        PlannedOp::Rank => Some("MidRankTransform"),
        PlannedOp::Blom => Some("BlomNormalScores"),
        PlannedOp::QuantileBins { .. } => Some("QuantileBinIndexer"),
        PlannedOp::BinOnehot { .. } => Some("QuantileOnehotEncoder"),
        PlannedOp::Hinge { .. } => Some("HingeBasis"),
        PlannedOp::WinsorClip { .. } => Some("WinsorLimiter"),
        PlannedOp::Cheb { .. } => None,
        PlannedOp::PolyInteract => Some("PairwiseProducts"),
        PlannedOp::RffCos { .. } => Some("RandomFourierCosineFeatures"),
        PlannedOp::RffSin { .. } => Some("RandomFourierSineFeatures"),
        PlannedOp::RandProj { .. } => Some("GaussianRandomProjection"),
        PlannedOp::LandmarkDist { .. } => Some("GaussianLandmarkFeatures"),
        PlannedOp::ObliqueHinge { .. } => Some("ObliqueHingeFeatures"),
        PlannedOp::RowStats => None,
        PlannedOp::PairProd { .. } => Some("CorrelationScreenedProducts"),
        PlannedOp::TargetBinEncode { .. } => Some("SmoothedTargetBinEncoder"),
        PlannedOp::Sigmoid
        | PlannedOp::Symlog { .. }
        | PlannedOp::Step { .. }
        | PlannedOp::NormRow
        | PlannedOp::Polar
        | PlannedOp::Lag { .. }
        | PlannedOp::Roll { .. }
        | PlannedOp::RowSpectral
        | PlannedOp::RowDiffStats
        | PlannedOp::MapExpr { .. }
        | PlannedOp::DeriveExpr { .. } => None,
        // Fitted state is the resolved frequency count.
        PlannedOp::FftMag { .. } => Some("usize"),
        PlannedOp::SvdProject { .. } => Some("PrincipalComponentProjection"),
        PlannedOp::Nystrom { .. } => Some("NystromLandmarkFeatures"),
        PlannedOp::SelectCorr { .. } => Some("ColumnSelect"),
        PlannedOp::Impute { .. } => Some("ZeroImputer"),
        PlannedOp::PairRatio { .. } => Some("RatioScreenedPairs"),
        PlannedOp::Spline { .. } => Some("NaturalCubicSpline"),
    }
}

/// One-sentence math description with a literature anchor, used in docs.
pub fn op_doc(op: &PlannedOp) -> String {
    match op {
        // B13: version-two sparse Fourier power bank.
        PlannedOp::RowVectorV2 {
            state: crate::kernel::row_vector::FittedStage::Goertzel(_),
        } => "computes, for each frozen frequency, the squared magnitude of the row's discrete Fourier sum with a second-order recurrence (Goertzel, 1958) and a sum-of-squares finalization.".into(),
        // B12: version-two FeatureWithinRow stage.
        PlannedOp::RowVectorV2 { .. } | PlannedOp::RowVectorInputV2 { .. } => "applies a width-bound feature-position FIR with frozen coefficient bits; clipped boundary windows reduce polynomial degree.".into(),
        PlannedOp::Standardize => "centers each column by its training mean and divides by the \
            standard deviation (1/n convention, floored at 1e-9), clamping the score to [-20, 20]."
            .to_string(),
        PlannedOp::RobustScale => "centers each column at its training median and divides by the \
            interquartile range, the outlier-resistant scaling of exploratory data analysis \
            (Tukey, 1977)."
            .to_string(),
        PlannedOp::Rank => "replaces each value by its mid-rank within the training column, i.e. \
            the average of the strict and non-strict ranks scaled to [0, 1], an empirical \
            distribution-function transform."
            .to_string(),
        PlannedOp::Blom => "maps mid-ranks to standard normal quantiles through the plotting \
            position (r + 1/8)/(n + 1/4), a normal-scores transform in the tradition of Blom \
            (1958), computed with Acklam's rational probit approximation."
            .to_string(),
        PlannedOp::QuantileBins { bins } => format!(
            "discretizes each column into {bins} equal-frequency bins at training quantiles and \
             reports the bin index scaled to [0, 1]."
        ),
        PlannedOp::BinOnehot { bins } => format!(
            "one-hot encodes {bins} equal-frequency quantile bins per column (an indicator basis)."
        ),
        PlannedOp::Hinge { knots } => format!(
            "expands each column into {knots} hinge functions max(x - t, 0) with knots at training \
             quantiles, the basis of multivariate adaptive regression splines (Friedman, 1991)."
        ),
        PlannedOp::WinsorClip { q } => format!(
            "winsorizes each column by clipping to its [{q}, 1-{q}] training quantiles, limiting \
             the leverage of extreme values (Tukey & McLaughlin, 1963).",
            q = crate::mathexpr::fmt_f64(*q)
        ),
        PlannedOp::Cheb { degree } => format!(
            "evaluates Chebyshev polynomials of the first kind T_1..T_{degree} on u = clip(2x - 1) \
             via the recurrence T_(k+1) = 2u T_k - T_(k-1)."
        ),
        PlannedOp::PolyInteract => "forms degree-two polynomial features: the square of every \
            column plus pairwise products, capped at 512 outputs."
            .to_string(),
        PlannedOp::RffCos { n, gamma, .. } => format!(
            "computes {n} random Fourier features sqrt(2/n)*cos(w.x + b) with Gaussian frequencies \
             scaled by sqrt(2*{g}) and uniform phases; such features approximate a shift-invariant \
             Gaussian kernel (Rahimi & Recht, 2007).",
            g = crate::mathexpr::fmt_f64(*gamma)
        ),
        PlannedOp::RffSin { n, gamma, .. } => format!(
            "computes {n} random Fourier features sqrt(2/n)*sin(w.x + b) with Gaussian frequencies \
             scaled by sqrt(2*{g}) and uniform phases, the sine counterpart of the Rahimi & Recht \
             (2007) construction.",
            g = crate::mathexpr::fmt_f64(*gamma)
        ),
        PlannedOp::RandProj { k, .. } => format!(
            "projects each row onto {k} random Gaussian directions scaled by 1/sqrt(k), preserving \
             geometry in expectation (Johnson & Lindenstrauss, 1984)."
        ),
        PlannedOp::LandmarkDist { k, gamma, .. } => format!(
            "measures Gaussian kernel similarity exp(-{g}*||x - l||^2) to {k} fixed random \
             landmarks, a landmark feature map in the spirit of Nystrom approximations \
             (Williams & Seeger, 2001).",
            g = crate::mathexpr::fmt_f64(*gamma)
        ),
        PlannedOp::ObliqueHinge { n, .. } => format!(
            "emits {n} oblique hinge units max(w.x - t, 0) whose random directions are scaled by \
             1/sqrt(p) and whose thresholds sit at the median training projection."
        ),
        PlannedOp::RowStats => "summarizes each row by ten distribution statistics: mean, \
            standard deviation, min, max, median, lower and upper quartiles, skewness, excess \
            kurtosis, and mean energy."
            .to_string(),
        PlannedOp::PairProd { topk } => format!(
            "multiplies pairs among the {topk} columns most correlated (absolute Pearson) with the \
             target, a screened interaction expansion."
        ),
        PlannedOp::TargetBinEncode { bins, prior } => format!(
            "replaces each value with the smoothed mean of the target inside its quantile bin \
             ({bins} bins, additive prior {p} toward the global mean), following Micci-Barreca \
             (2001).",
            p = crate::mathexpr::fmt_f64(*prior)
        ),
        PlannedOp::Sigmoid => "squashes every value through the logistic function \
            1/(1 + exp(-v)) with the argument clipped to [-30, 30]."
            .to_string(),
        PlannedOp::Step { t } => format!(
            "replaces each value with the hard indicator 1.0 when the value exceeds the threshold \
             {t} and 0.0 otherwise (a Heaviside step evaluated as an explicit conditional).",
            t = crate::mathexpr::fmt_f64(*t)
        ),
        PlannedOp::NormRow => "scales each row to unit Euclidean length (the norm is floored at \
            1e-9)."
            .to_string(),
        PlannedOp::Polar => "rewrites each consecutive column pair (a, b) in polar coordinates: \
            radius sqrt(a^2 + b^2) then angle atan2(b, a); an odd trailing column passes through \
            unchanged."
            .to_string(),
        PlannedOp::Lag { k } => format!(
            "shifts each row {k} columns to the right, filling vacated positions with zeros (a \
             within-row lag)."
        ),
        PlannedOp::Roll { k } => {
            format!("rotates each row {k} columns to the right (a circular shift).")
        }
        PlannedOp::FftMag { k } => format!(
            "computes magnitudes of the discrete Fourier transform of each row at frequencies \
             1..{k} (capped at half the width), scaled by 1/d — the direct O(d^2) sum for exact \
             reproducibility."
        ),
        PlannedOp::RowSpectral => "summarizes the spectrum of each row over DFT frequencies \
            1..d/2 by spectral entropy, first and second spectral moments, peak frequency, and \
            peak magnitude."
            .to_string(),
        PlannedOp::RowDiffStats => "summarizes the first differences of each row by mean, \
            standard deviation, min, max, mean absolute change, sign-change rate, lag-1 \
            autocorrelation of the original row, and mean squared change."
            .to_string(),
        PlannedOp::SvdProject { k, iters, .. } => format!(
            "projects mean-centered rows onto the top {k} principal directions found by {iters} \
             rounds of deterministic power iteration with Gram-Schmidt deflation (Pearson, 1901; \
             Golub & Van Loan, 2013)."
        ),
        PlannedOp::Nystrom { k, gamma, .. } => format!(
            "measures Gaussian kernel similarity exp(-{g}*||x - l||^2) to {k} training rows \
             chosen by a seeded partial Fisher-Yates shuffle, the Nystrom landmark construction \
             (Williams & Seeger, 2001).",
            g = crate::mathexpr::fmt_f64(*gamma)
        ),
        PlannedOp::MapExpr { source, .. } => format!(
            "applies the scalar formula f(x, row) = {source} to every cell, where x is the cell \
             value and row is the full row, with guarded arithmetic."
        ),
        PlannedOp::DeriveExpr { source, .. } => format!(
            "reduces each row to the single derived column g(row) = {source}, with guarded \
             arithmetic."
        ),
        PlannedOp::Symlog { c } => format!(
            "compresses heavy tails through the signed shifted logarithm asinh(x / {c}) = \
             sign(u) * ln(|u| + sqrt(u^2 + 1)) with u = x / {c}, a smooth monotone transform \
             that is linear near zero, logarithmic in the tails, and extrapolates beyond the \
             training range.",
            c = crate::mathexpr::fmt_f64(*c)
        ),
        PlannedOp::SelectCorr { k, by_var } => {
            if *by_var {
                format!(
                    "keeps the {k} columns with the largest training variance (ties to the \
                     lower index, original column order preserved), an unsupervised screen \
                     that narrows wide inputs."
                )
            } else {
                format!(
                    "keeps the {k} columns with the largest absolute Pearson correlation to \
                     the training target (ties to the lower index, original column order \
                     preserved), the sure-independence-screening step of Fan & Lv (2008)."
                )
            }
        }
        PlannedOp::Impute {
            strategy,
            add_indicator,
            ..
        } => {
            let fill = match strategy {
                0 => "the mean of the column's non-zero training values",
                2 => "a fixed constant",
                _ => "the median of the column's non-zero training values",
            };
            if *add_indicator {
                format!(
                    "treats exact-zero cells as missing and replaces them with {fill}, \
                     appending one 0/1 missingness indicator per column that contained a \
                     zero in training (Rubin, 1976)."
                )
            } else {
                format!(
                    "treats exact-zero cells as missing and replaces them with {fill} \
                     (Rubin, 1976)."
                )
            }
        }
        PlannedOp::PairRatio { topk } => format!(
            "forms the {topk} regularized ratios x_a * x_b / (x_b^2 + eps) with the strongest \
             absolute Pearson correlation to the training target, screening pairs over the 32 \
             most target-correlated columns; eps = max(1e-12, (0.01 * training std of x_b)^2) \
             keeps every ratio finite and smooth."
        ),
        PlannedOp::Spline { knots } => format!(
            "expands each column in the natural cubic spline basis with {knots} knots at the \
             interior training quantiles i/({knots}+1) (Hastie, Tibshirani & Friedman, ESL \
             §5.2.1): the raw value plus {m} truncated-power basis functions constrained to \
             extrapolate linearly beyond the boundary knots; a near-constant column falls back \
             to its raw value.",
            m = knots - 2
        ),
    }
}

/// Short phrase for titles.
pub fn op_short(op: &PlannedOp) -> &'static str {
    match op {
        // B13: version-two sparse Fourier power bank.
        PlannedOp::RowVectorV2 {
            state: crate::kernel::row_vector::FittedStage::Goertzel(_),
        } => "Fourier power bank",
        // B12: version-two FeatureWithinRow stage.
        PlannedOp::RowVectorV2 { .. } | PlannedOp::RowVectorInputV2 { .. } => {
            "feature-position filter"
        }
        PlannedOp::Standardize => "standard scaling",
        PlannedOp::RobustScale => "robust scaling",
        PlannedOp::Rank => "mid-rank transform",
        PlannedOp::Blom => "normal scores",
        PlannedOp::QuantileBins { .. } => "quantile binning",
        PlannedOp::BinOnehot { .. } => "quantile one-hot encoding",
        PlannedOp::Hinge { .. } => "hinge basis",
        PlannedOp::WinsorClip { .. } => "winsorizing",
        PlannedOp::Cheb { .. } => "Chebyshev basis",
        PlannedOp::PolyInteract => "pairwise products",
        PlannedOp::RffCos { .. } => "random Fourier cosine features",
        PlannedOp::RffSin { .. } => "random Fourier sine features",
        PlannedOp::RandProj { .. } => "Gaussian random projection",
        PlannedOp::LandmarkDist { .. } => "Gaussian landmark features",
        PlannedOp::ObliqueHinge { .. } => "oblique hinge features",
        PlannedOp::RowStats => "row summary statistics",
        PlannedOp::PairProd { .. } => "correlation-screened products",
        PlannedOp::TargetBinEncode { .. } => "smoothed target bin encoding",
        PlannedOp::Sigmoid => "logistic squash",
        PlannedOp::Step { .. } => "step indicator",
        PlannedOp::NormRow => "unit row scaling",
        PlannedOp::Polar => "polar coordinates",
        PlannedOp::Lag { .. } => "column shift",
        PlannedOp::Roll { .. } => "column rotation",
        PlannedOp::FftMag { .. } => "Fourier magnitudes",
        PlannedOp::RowSpectral => "row spectral summary",
        PlannedOp::RowDiffStats => "row difference statistics",
        PlannedOp::SvdProject { .. } => "principal component projection",
        PlannedOp::Nystrom { .. } => "Nystrom landmark features",
        PlannedOp::MapExpr { .. } => "an elementwise formula",
        PlannedOp::DeriveExpr { .. } => "a derived column",
        PlannedOp::Symlog { .. } => "signed log compression",
        PlannedOp::SelectCorr { .. } => "column screening",
        PlannedOp::Impute { .. } => "zero imputation",
        PlannedOp::PairRatio { .. } => "ratio features",
        PlannedOp::Spline { .. } => "natural cubic splines",
    }
}

pub fn head_fn_name(head: &PlannedHead) -> &'static str {
    match head {
        PlannedHead::Ridge { .. } => "ridge_regression",
        PlannedHead::RidgeGcv => "gcv_ridge_regression",
        PlannedHead::ElasticNet { .. } => "elastic_net",
        PlannedHead::Gbm(_) => "gradient_boosted_trees",
        PlannedHead::Blend { .. } => "blended_regression",
        PlannedHead::HistGbm2 { .. } => "leafwise_boosted_trees",
        PlannedHead::Krr { .. } => "kernel_ridge_regression",
        PlannedHead::Gpr { .. } => "gaussian_process_ard_regression",
        PlannedHead::GprKernel { .. } => "gaussian_process_composite_kernel_regression",
        PlannedHead::Ksmooth { .. } => "nadaraya_watson_regression",
        PlannedHead::Stack => "greedy_ensemble_selection",
        PlannedHead::Portfolio(_) => "internally_validated_ensemble",
    }
}

/// Emitted-artifact tag for one kernel primitive (used in the kernel-terms
/// literal both emitters bake into the composite-GP head).
pub fn kernel_prim_tag(prim: KernelPrim) -> &'static str {
    match prim {
        KernelPrim::Rbf => "squared_exponential",
        KernelPrim::Rq => "rational_quadratic",
        KernelPrim::Periodic => "periodic",
        KernelPrim::Linear => "linear",
    }
}

/// One-sentence description of a kernel primitive for the docs.
fn kernel_prim_doc(prim: KernelPrim) -> &'static str {
    match prim {
        KernelPrim::Rbf => {
            "a squared-exponential factor with per-feature \
            automatic-relevance lengthscales"
        }
        KernelPrim::Rq => {
            "a rational-quadratic factor (1 + r^2/(2 alpha))^-alpha \
            with per-feature lengthscales and a learned mixing exponent alpha, a \
            scale mixture of squared exponentials"
        }
        KernelPrim::Periodic => {
            "a periodic factor exp(-2 sum_d sin^2(pi (x_d - \
            x'_d)/period) / l^2) with a learned shared period and lengthscale \
            (MacKay, 1998)"
        }
        KernelPrim::Linear => {
            "a homogeneous linear factor (x . x')/p, giving a \
            Bayesian-linear-regression component"
        }
    }
}

pub fn head_doc(head: &PlannedHead) -> String {
    match head {
        PlannedHead::Ridge { lambda_frac } => format!(
            "ridge regression (Hoerl & Kennard, 1970) on standardized columns with penalty \
             max({lf}*n, 1e-3); the symmetric positive-definite normal equations are solved by a \
             Cholesky factorization and predictions are clipped to [-60, 60].",
            lf = crate::mathexpr::fmt_f64(*lambda_frac)
        ),
        PlannedHead::RidgeGcv => "ridge regression on standardized columns whose penalty is \
             chosen by generalized cross-validation, GCV(lambda) = mean squared residual / \
             max(1 - tr(S_lambda)/n, 1e-3)^2 (Golub, Heath & Wahba, 1979), over a fixed 25-point \
             logarithmic grid from 1e-5 to 1e3; when rows and columns both exceed 384 the 384 \
             columns most correlated with the target are kept (a train-only screen), one \
             cyclic-Jacobi eigendecomposition of the smaller of the p x p standardized Gram \
             matrix and the n x n inner-product matrix (they share their nonzero spectrum) \
             scores every grid penalty in closed form, the winning penalty is refit on all \
             training rows, and predictions are clipped to [-60, 60]."
            .to_string(),
        PlannedHead::ElasticNet { alpha, l1_ratio, sweeps } => format!(
            "elastic net regression (Zou & Hastie, 2005) with alpha={a}, l1_ratio={l}, fit by at \
             most {sweeps} sweeps of cyclic coordinate descent with soft-thresholding (Friedman, \
             Hastie & Tibshirani, 2010); predictions are clipped to [-60, 60].",
            a = crate::mathexpr::fmt_f64(*alpha),
            l = crate::mathexpr::fmt_f64(*l1_ratio)
        ),
        PlannedHead::Gbm(g) => format!(
            "gradient boosted regression trees (Friedman, 2001) with histogram splits on training \
             quantile bins and second-order leaf values -g/(h + lambda) as in modern boosting \
             systems (Chen & Guestrin, 2016): {rounds} rounds, depth {depth}, learning rate {lr}, \
             lambda {lam}, {bins} bins, min leaf {ml}.",
            rounds = g.rounds,
            depth = g.depth,
            lr = crate::mathexpr::fmt_f64(g.lr),
            lam = crate::mathexpr::fmt_f64(g.lambda),
            bins = g.bins,
            ml = g.min_leaf
        ),
        PlannedHead::Blend { lambda_frac, gbm } => format!(
            "a convex blend w*linear + (1-w)*trees of ridge regression (lambda fraction {lf}) and \
             gradient boosted trees ({rounds} rounds, depth {depth}); w is picked from \
             {{0, 0.25, 0.5, 0.75, 1}} on an internal 80/20 holdout by root mean squared error and \
             both members are then refit on all training rows.",
            lf = crate::mathexpr::fmt_f64(*lambda_frac),
            rounds = gbm.rounds,
            depth = gbm.depth
        ),
        PlannedHead::HistGbm2 {
            leaves,
            lr,
            bags,
            rounds,
            row_subsample,
            col_subsample,
            min_leaf,
            val_pct,
            val_shuffle,
            huber_delta,
            stop_mae,
        } => {
            let col_doc = if *col_subsample < 1.0 {
                format!(
                    " and a column-sampling share of {}",
                    crate::mathexpr::fmt_f64(*col_subsample)
                )
            } else {
                String::new()
            };
            let loss_doc = if *huber_delta > 0.0 {
                format!(
                    " Gradients are the residuals clipped at {} — the Huber loss (Huber, 1964).",
                    crate::mathexpr::fmt_f64(*huber_delta)
                )
            } else {
                String::new()
            };
            let stop_metric = if *stop_mae {
                "mean absolute error"
            } else {
                "root mean squared error"
            };
            let split_kind = if *val_shuffle {
                "rows chosen by a seeded shuffle"
            } else {
                "the contiguous tail rows"
            };
            format!(
                "gradient boosted regression trees (Friedman, 2001) grown leaf-wise / best-first \
                 to at most {leaves} leaves with exact split search over pre-sorted feature \
                 columns, in the manner of LightGBM (Ke et al., 2017); leaf values are the \
                 second-order step -g/(h + 1) (Chen & Guestrin, 2016), learning rate {lr}, \
                 minimum leaf size {min_leaf}, with a per-tree row-sampling share of \
                 {rs}{col_doc}.{loss_doc} {bags} bagged replicate(s) (Breiman, 1996) each stop \
                 early by {stop_metric} on an internal {val_pct}% holdout ({split_kind}; \
                 patience 50 rounds, at most {rounds}) and predictions average the replicates.",
                lr = crate::mathexpr::fmt_f64(*lr),
                rs = crate::mathexpr::fmt_f64(*row_subsample)
            )
        }
        PlannedHead::Krr { m } => format!(
            "kernel ridge regression (Saunders, Gammerman & Vovk, 1998) with a Gaussian kernel on \
             features scaled by the inverse training standard deviation; the kernel width gamma \
             (5 values scaled by 1/d) and ridge penalty lambda (5 values) are chosen by root mean \
             squared error on a contiguous 80/20 holdout using at most 256 seeded kernel centers, \
             the final coefficient system (K + lambda I) a = y over at most {m} seeded centers is \
             solved by Cholesky factorization, and predictions k(x)^T a are clipped to [-60, 60]."
        ),
        PlannedHead::Gpr { m, steps } => format!(
            "Gaussian-process regression (Rasmussen & Williams, 2006) with a squared-exponential \
             automatic-relevance-determination kernel sf2 * exp(-0.5 * sum_d (x_d - x'_d)^2 / ls_d^2) \
             on features scaled by the inverse training standard deviation; the per-feature \
             lengthscales ls_d, signal variance sf2, and noise variance sn2 are learned by {steps} \
             steps of gradient ascent on the log marginal likelihood, whose gradient \
             0.5 * trace((a a^T - Ky^-1) dKy/dtheta) with a = Ky^-1 y is evaluated with one \
             Cholesky-based inverse per step on at most 256 seeded rows; the predictive coefficients \
             are then solved on up to {m} seeded inducing rows and predictions k(x)^T a are clipped \
             to [-60, 60]."
        ),
        PlannedHead::GprKernel { m, steps, source, spec } => {
            let mut seen: Vec<KernelPrim> = Vec::new();
            for term in spec {
                for prim in term {
                    if !seen.contains(prim) {
                        seen.push(*prim);
                    }
                }
            }
            let prim_docs: Vec<&str> = seen.iter().map(|p| kernel_prim_doc(*p)).collect();
            format!(
                "Gaussian-process regression (Rasmussen & Williams, 2006) whose covariance is the \
                 composed kernel \"{source}\" — a sum of products of primitives in the spirit of \
                 kernel structure search (Duvenaud et al., 2013): {prims}. Each sum term carries \
                 a learned amplitude; every factor's shape parameters and the noise variance are \
                 learned on features scaled by the inverse training standard deviation via \
                 {steps} steps of gradient ascent on the log marginal likelihood, whose gradient \
                 0.5 * trace((a a^T - Ky^-1) dKy/dtheta) with a = Ky^-1 y is evaluated with one \
                 Cholesky-based inverse per step on at most 256 seeded rows; the predictive \
                 coefficients are then solved on up to {m} seeded inducing rows and predictions \
                 k(x)^T a are clipped to [-60, 60].",
                prims = prim_docs.join("; ")
            )
        }
        PlannedHead::Ksmooth { k, gamma } => format!(
            "Nadaraya-Watson kernel regression (Nadaraya, 1964; Watson, 1964): each prediction \
             is the Gaussian-weighted average of the targets of the {k} nearest training rows in \
             standardized feature space, with weights exp(-{g} * (d^2 - d^2_min)/p) over squared \
             Euclidean distances (ties broken by the row index); subtracting the nearest \
             distance d^2_min keeps the largest weight at exactly 1 so the denominator never \
             vanishes, and predictions are clipped to [-60, 60].",
            g = crate::mathexpr::fmt_f64(*gamma)
        ),
        PlannedHead::Stack => format!(
            "greedy ensemble selection with replacement (Caruana, Niculescu-Mizil, Crew & Ksikes, \
             2004) over a fixed five-model library: ridge regression at two penalties, an elastic \
             net, leaf-wise boosted trees ({leaves} leaves, learning rate {lr}, {bags} bagged \
             replicates), and Gaussian kernel ridge regression (at most {m} centers). Out-of-fold \
             predictions from contiguous 5-fold splits guide up to 20 greedy additions maximizing \
             the R^2 of the running mean; final weights are selection frequencies and each \
             selected model is refit on all training rows.",
            leaves = crate::boost2::DEFAULT_LEAVES,
            lr = crate::mathexpr::fmt_f64(crate::boost2::DEFAULT_LR),
            bags = crate::boost2::DEFAULT_BAGS,
            m = crate::krr::DEFAULT_M
        ),
        PlannedHead::Portfolio(p) => format!(
            "an internally validated ensemble of {n} component pipelines, each fitted on the \
             training rows and measured with held-out predictions from seeded shuffled \
             {folds}-fold splits (repeated on smaller samples for stability), every component \
             refit inside each fold so no fitted statistic sees its validation rows. Component \
             activation depends only on the training sample's row count, column count and \
             duplicate-row share; per-fold predictions are clipped to the fold's padded target \
             range; the components are then combined by a non-negative ridge blend of their \
             deviations from the training mean, solved by exact cyclic coordinate descent (at \
             most 500 sweeps) with a shrinkage penalty that grows with the component count and \
             shrinks with the sample size, falling back to the single best-measured component \
             when the blend loses. Components with meaningful weight are refit on all training \
             rows, a refit whose training error exceeds 1.05 standard deviations of the target \
             is replaced by the mean predictor, and every prediction is clipped to the padded \
             training target range.",
            n = p.members.iter().filter(|m| !m.hidden).count(),
            folds = p.folds
        ),
    }
}

pub fn head_short(head: &PlannedHead) -> &'static str {
    match head {
        PlannedHead::Ridge { .. } => "ridge regression",
        PlannedHead::RidgeGcv => "GCV-tuned ridge regression",
        PlannedHead::ElasticNet { .. } => "elastic net regression",
        PlannedHead::Gbm(_) => "gradient boosted trees",
        PlannedHead::Blend { .. } => "a ridge / boosted-trees blend",
        PlannedHead::HistGbm2 { .. } => "leaf-wise boosted trees",
        PlannedHead::Krr { .. } => "kernel ridge regression",
        PlannedHead::Gpr { .. } => "Gaussian-process regression",
        PlannedHead::GprKernel { .. } => "Gaussian-process regression with a composed kernel",
        PlannedHead::Ksmooth { .. } => "a nearest-neighbor kernel smoother",
        PlannedHead::Stack => "a greedy model ensemble",
        PlannedHead::Portfolio(_) => "an internally validated model ensemble",
    }
}

/// Stable per-kind key: also the base of the emitted target-transform
/// function names.
pub fn target_fn_name(kind: TargetKind) -> &'static str {
    match kind {
        TargetKind::RankGauss => "normal_scores_target",
        TargetKind::Log1p => "shifted_log_target",
        TargetKind::Winsor => "winsorized_target",
        TargetKind::ZScore => "standardized_target",
    }
}

/// True for kinds whose inverse carries its own train-range prediction clamp
/// instead of the historical ±60 wrap (which silently destroys any target
/// whose scale exceeds it).
pub fn target_range_aware(kind: TargetKind) -> bool {
    matches!(kind, TargetKind::ZScore)
}

/// Rust fitted-state struct name (None for the stateless kind).
pub fn target_struct_name(kind: TargetKind) -> Option<&'static str> {
    match kind {
        TargetKind::RankGauss => Some("NormalScoresTarget"),
        TargetKind::Log1p => None,
        TargetKind::Winsor => Some("WinsorizedTarget"),
        TargetKind::ZScore => Some("StandardizedTarget"),
    }
}

/// One-sentence math description of a target transform, used in docs.
pub fn target_doc(target: &PlannedTarget) -> String {
    match target.kind {
        TargetKind::RankGauss => "replaces each training-target value by the standard normal \
            quantile of its mid-rank (Acklam's rational probit approximation), a normal-scores \
            transform; the inverse interpolates linearly through the fitted table of distinct \
            (score, value) pairs, clamped to the extreme training values."
            .to_string(),
        TargetKind::Log1p => "maps each training-target value through the shifted logarithm \
            ln(1 + max(y, -0.999999)); the inverse is exp(z) - 1 with the exponent clipped to \
            [-745, 700]."
            .to_string(),
        TargetKind::Winsor => format!(
            "winsorizes the training target by clipping to its [{q}, 1-{q}] quantiles, limiting \
             the leverage of extreme targets (Tukey & McLaughlin, 1963); the inverse is the \
             identity, so predictions stay drawn toward the bulk of the target.",
            q = crate::mathexpr::fmt_f64(target.q)
        ),
        TargetKind::ZScore => "standardizes the training target to zero mean and unit variance \
            (standard deviation floored at 1e-9); the inverse maps predictions back through \
            z * sigma + mu and clamps them to the training-target range widened by 5% or the \
            [-60, 60] window, whichever is wider."
            .to_string(),
    }
}

/// Short phrase for titles.
pub fn target_short(kind: TargetKind) -> &'static str {
    match kind {
        TargetKind::RankGauss => "a normal-scores target transform",
        TargetKind::Log1p => "a shifted-log target transform",
        TargetKind::Winsor => "a winsorized target",
        TargetKind::ZScore => "a standardized (z-score) target transform",
    }
}

// ------------------------------- helper set --------------------------------

/// Shared helper snippets, ordered by declaration (= emission order).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Helper {
    Clamp,
    Fmin,
    Fmax,
    SafeSin,
    SafeCos,
    PowGuard,
    MathDiv,
    MathLog1p,
    MathSigmoid,
    MathClip,
    Asinh,
    MathExp,
    MathLog,
    Erf,
    Sign,
    MathFloor,
    RoundHalfAway,
    RowMean,
    RowSum,
    RowL1,
    RowL2,
    RowMin,
    RowMax,
    RowStd,
    Pcg32,
    RngUniform,
    RngRange,
    RngGaussian,
    StageRng,
    Sanitize,
    FinishStage,
    SortedColumn,
    Quantile,
    LowerBound,
    UpperBound,
    MidRank,
    Probit,
    QuantileEdges,
    Dot,
    Normalize,
    RbfRow,
    DftMags,
    MeanOf,
    Pearson,
    Rmse,
    RSquared,
    StdFit,
    CholeskySolve,
    JacobiEigen,
    SoftThreshold,
    TreeCore,
}

fn helper_deps(h: Helper) -> &'static [Helper] {
    match h {
        Helper::MathSigmoid => &[Helper::Clamp],
        Helper::MathExp => &[Helper::Clamp],
        Helper::MathClip => &[Helper::Clamp, Helper::Fmin, Helper::Fmax],
        Helper::RowMin => &[Helper::Fmin],
        Helper::RowMax => &[Helper::Fmax],
        Helper::RowStd => &[Helper::RowMean],
        Helper::RngUniform => &[Helper::Pcg32],
        Helper::RngRange => &[Helper::RngUniform],
        Helper::RngGaussian => &[Helper::RngUniform],
        Helper::StageRng => &[Helper::Pcg32],
        Helper::MidRank => &[Helper::LowerBound, Helper::UpperBound],
        Helper::Probit => &[Helper::Clamp],
        Helper::QuantileEdges => &[Helper::SortedColumn, Helper::Quantile],
        Helper::TreeCore => &[Helper::LowerBound, Helper::MeanOf],
        _ => &[],
    }
}

fn add_helper(set: &mut BTreeSet<Helper>, h: Helper) {
    if set.insert(h) {
        for d in helper_deps(h) {
            add_helper(set, *d);
        }
    }
}

fn op_helpers(op: &PlannedOp) -> &'static [Helper] {
    match op {
        // B12: version-two FeatureWithinRow stage.
        PlannedOp::RowVectorV2 { .. } | PlannedOp::RowVectorInputV2 { .. } => &[],
        PlannedOp::Standardize => &[Helper::StdFit, Helper::Clamp],
        PlannedOp::RobustScale => &[Helper::SortedColumn, Helper::Quantile],
        PlannedOp::Rank => &[Helper::SortedColumn, Helper::MidRank],
        PlannedOp::Blom => &[Helper::SortedColumn, Helper::MidRank, Helper::Probit],
        PlannedOp::QuantileBins { .. } => &[Helper::QuantileEdges, Helper::LowerBound],
        PlannedOp::BinOnehot { .. } => &[Helper::QuantileEdges, Helper::LowerBound],
        PlannedOp::Hinge { .. } => &[Helper::SortedColumn, Helper::Quantile],
        PlannedOp::WinsorClip { .. } => &[Helper::SortedColumn, Helper::Quantile, Helper::Clamp],
        PlannedOp::Cheb { .. } => &[Helper::Clamp],
        PlannedOp::PolyInteract => &[],
        PlannedOp::RffCos { .. } => &[
            Helper::StageRng,
            Helper::RngGaussian,
            Helper::RngRange,
            Helper::Dot,
            Helper::SafeCos,
        ],
        PlannedOp::RffSin { .. } => &[
            Helper::StageRng,
            Helper::RngGaussian,
            Helper::RngRange,
            Helper::Dot,
            Helper::SafeSin,
        ],
        PlannedOp::RandProj { .. } => &[Helper::StageRng, Helper::RngGaussian, Helper::Dot],
        PlannedOp::LandmarkDist { .. } => &[Helper::StageRng, Helper::RngRange, Helper::RbfRow],
        PlannedOp::ObliqueHinge { .. } => &[
            Helper::StageRng,
            Helper::RngGaussian,
            Helper::Dot,
            Helper::Quantile,
        ],
        PlannedOp::RowStats => &[Helper::Quantile],
        PlannedOp::PairProd { .. } => &[Helper::Pearson],
        PlannedOp::TargetBinEncode { .. } => &[Helper::QuantileEdges, Helper::LowerBound],
        PlannedOp::Sigmoid => &[Helper::MathSigmoid],
        PlannedOp::Step { .. } => &[],
        PlannedOp::NormRow => &[],
        PlannedOp::Polar => &[],
        PlannedOp::Lag { .. } => &[],
        PlannedOp::Roll { .. } => &[],
        PlannedOp::FftMag { .. } => &[Helper::DftMags],
        PlannedOp::RowSpectral => &[Helper::DftMags],
        PlannedOp::RowDiffStats => &[Helper::Pearson],
        PlannedOp::SvdProject { .. } => &[
            Helper::StageRng,
            Helper::RngGaussian,
            Helper::Dot,
            Helper::Normalize,
        ],
        PlannedOp::Nystrom { .. } => &[Helper::StageRng, Helper::RbfRow],
        PlannedOp::Symlog { .. } => &[Helper::Asinh],
        // Both scoring branches are emitted regardless of the runtime flag,
        // so the pearson helper is always required.
        PlannedOp::SelectCorr { .. } => &[Helper::Pearson],
        // All three strategy branches are emitted; median needs the quantile.
        PlannedOp::Impute { .. } => &[Helper::Quantile],
        // Pair screening + fmax for the denominator guard floor.
        PlannedOp::PairRatio { .. } => &[Helper::Pearson, Helper::Fmax],
        // Same fit machinery as hinge.
        PlannedOp::Spline { .. } => &[Helper::SortedColumn, Helper::Quantile],
        // Expression-dependent helpers are added separately in helper_set().
        PlannedOp::MapExpr { .. } | PlannedOp::DeriveExpr { .. } => &[],
    }
}

fn head_helpers(head: &PlannedHead) -> &'static [Helper] {
    match head {
        PlannedHead::Ridge { .. } => &[Helper::StdFit, Helper::CholeskySolve, Helper::Clamp],
        PlannedHead::RidgeGcv => &[Helper::StdFit, Helper::JacobiEigen, Helper::Clamp],
        PlannedHead::ElasticNet { .. } => &[Helper::StdFit, Helper::SoftThreshold, Helper::Clamp],
        PlannedHead::Gbm(_) => &[Helper::TreeCore],
        PlannedHead::Blend { .. } => &[
            Helper::StdFit,
            Helper::CholeskySolve,
            Helper::Clamp,
            Helper::TreeCore,
            Helper::Rmse,
        ],
        PlannedHead::HistGbm2 { .. } => &[
            Helper::RngUniform,
            Helper::Rmse,
            Helper::MeanOf,
            Helper::Clamp,
        ],
        PlannedHead::Krr { .. } => &[
            Helper::StdFit,
            Helper::CholeskySolve,
            Helper::Clamp,
            Helper::Rmse,
            Helper::Pcg32,
        ],
        PlannedHead::Gpr { .. } => &[
            Helper::StdFit,
            Helper::CholeskySolve,
            Helper::Clamp,
            Helper::Quantile,
            Helper::Pcg32,
        ],
        PlannedHead::GprKernel { .. } => &[
            Helper::StdFit,
            Helper::CholeskySolve,
            Helper::Clamp,
            Helper::Quantile,
            Helper::Pcg32,
        ],
        PlannedHead::Ksmooth { .. } => &[Helper::StdFit, Helper::Clamp],
        // Union of every library member plus the greedy-selection metric.
        PlannedHead::Stack => &[
            Helper::StdFit,
            Helper::CholeskySolve,
            Helper::Clamp,
            Helper::SoftThreshold,
            Helper::RngUniform,
            Helper::Rmse,
            Helper::RSquared,
            Helper::MeanOf,
            Helper::Pcg32,
        ],
        // The portfolio's own machinery: the fold-shuffle PCG stream plus the
        // wide-data projection view (an svd_project twin). Member ops/heads
        // add their helpers in helper_set (the set is member-dependent).
        PlannedHead::Portfolio(_) => &[
            Helper::Pcg32,
            Helper::Sanitize,
            Helper::FinishStage,
            Helper::StageRng,
            Helper::RngGaussian,
            Helper::Dot,
            Helper::Normalize,
        ],
    }
}

fn target_helpers(kind: TargetKind) -> &'static [Helper] {
    match kind {
        // fit sorts y and builds the probit(mid-rank) table; the inverse
        // binary-searches the fitted scores.
        TargetKind::RankGauss => &[
            Helper::MidRank,
            Helper::Probit,
            Helper::LowerBound,
            Helper::Clamp,
        ],
        // forward floors the shifted-log argument; the inverse clips the exponent.
        TargetKind::Log1p => &[Helper::Fmax, Helper::Clamp],
        TargetKind::Winsor => &[Helper::Quantile, Helper::Clamp],
        // fit tracks the target range with fmin/fmax and floors sigma with
        // fmax; the inverse applies the train-range clamp.
        TargetKind::ZScore => &[Helper::Fmin, Helper::Fmax, Helper::Clamp],
    }
}

fn expr_helpers(f: &ExprFeatures, set: &mut BTreeSet<Helper>) {
    if f.uses_div {
        add_helper(set, Helper::MathDiv);
    }
    if f.uses_pow {
        add_helper(set, Helper::PowGuard);
        add_helper(set, Helper::Clamp);
    }
    for func in &f.funcs {
        match func {
            MathFunc::Sqrt => add_helper(set, Helper::Fmax),
            MathFunc::Log1p => add_helper(set, Helper::MathLog1p),
            MathFunc::Sin => add_helper(set, Helper::SafeSin),
            MathFunc::Cos => add_helper(set, Helper::SafeCos),
            MathFunc::Sigmoid => add_helper(set, Helper::MathSigmoid),
            MathFunc::Min => add_helper(set, Helper::Fmin),
            MathFunc::Max => add_helper(set, Helper::Fmax),
            MathFunc::Clip => add_helper(set, Helper::MathClip),
            MathFunc::Exp => add_helper(set, Helper::MathExp),
            MathFunc::Log => add_helper(set, Helper::MathLog),
            MathFunc::Asinh => add_helper(set, Helper::Asinh),
            MathFunc::Erf => add_helper(set, Helper::Erf),
            MathFunc::Sign => add_helper(set, Helper::Sign),
            MathFunc::Floor => add_helper(set, Helper::MathFloor),
            MathFunc::Round => add_helper(set, Helper::RoundHalfAway),
            // step() emits an inline conditional and atan() maps to the
            // native call in both languages; neither needs a helper.
            MathFunc::Abs | MathFunc::Tanh | MathFunc::Step | MathFunc::Atan => {}
        }
    }
    for var in &f.vars {
        match var {
            MathVar::RowMean => add_helper(set, Helper::RowMean),
            MathVar::RowSum => add_helper(set, Helper::RowSum),
            MathVar::RowL1 => add_helper(set, Helper::RowL1),
            MathVar::RowL2 => add_helper(set, Helper::RowL2),
            MathVar::RowMin => add_helper(set, Helper::RowMin),
            MathVar::RowMax => add_helper(set, Helper::RowMax),
            MathVar::RowStd => add_helper(set, Helper::RowStd),
            MathVar::X | MathVar::Col(_) => {}
        }
    }
}

pub fn helper_set(plan: &Plan) -> BTreeSet<Helper> {
    let mut set = BTreeSet::new();
    match plan {
        Plan::Pipeline {
            target,
            stages,
            head,
        } => {
            add_helper(&mut set, Helper::Sanitize);
            add_helper(&mut set, Helper::FinishStage);
            if let Some(target) = target {
                for h in target_helpers(target.kind) {
                    add_helper(&mut set, *h);
                }
            }
            for stage in stages {
                for h in op_helpers(&stage.op) {
                    add_helper(&mut set, *h);
                }
                match &stage.op {
                    PlannedOp::MapExpr { expr, .. } => {
                        // x is defined for map, so the whole expression runs.
                        add_helper(&mut set, Helper::Clamp);
                        expr_helpers(&expr.features(), &mut set);
                    }
                    PlannedOp::DeriveExpr { expr, .. } => {
                        let features = expr.features();
                        if !features.uses_x {
                            add_helper(&mut set, Helper::Clamp);
                            expr_helpers(&features, &mut set);
                        }
                    }
                    _ => {}
                }
            }
            for h in head_helpers(head) {
                add_helper(&mut set, *h);
            }
            // Portfolio members bring their own op/head helper needs.
            if let PlannedHead::Portfolio(p) = head {
                for member in &p.members {
                    if let PlannedMemberKind::Pipeline { stages, head } = &member.kind {
                        for stage in stages {
                            for h in op_helpers(&stage.op) {
                                add_helper(&mut set, *h);
                            }
                        }
                        for h in head_helpers(head) {
                            add_helper(&mut set, *h);
                        }
                    }
                }
            }
        }
        Plan::Symbolic { expr, .. } => {
            add_helper(&mut set, Helper::Clamp);
            let features = expr.features();
            if !features.uses_x {
                expr_helpers(&features, &mut set);
            }
        }
    }
    set
}

// -------------------------------- output arena -----------------------------

/// Single-buffer section accumulator. Sections are appended with a fixed
/// separator, each trimmed of trailing whitespace, into one pre-reserved
/// `String`. This replaces the old `Vec<String>` + `join`, which allocated a
/// temporary owned+trimmed copy per section and then copied the whole program
/// a second time at join time; here every section is copied exactly once.
pub struct Arena {
    buf: String,
    sep: &'static str,
    first: bool,
}

impl Arena {
    pub fn new(sep: &'static str, capacity: usize) -> Arena {
        Arena {
            buf: String::with_capacity(capacity),
            sep,
            first: true,
        }
    }

    /// Append one section (its trailing whitespace trimmed, matching the prior
    /// per-section `trim_end()`), preceded by the separator except for the first.
    pub fn push(&mut self, section: &str) {
        if !self.first {
            self.buf.push_str(self.sep);
        }
        self.first = false;
        self.buf.push_str(section.trim_end());
    }

    /// Finalize with the trailing newline the emitters always appended.
    pub fn finish(mut self) -> String {
        self.buf.push('\n');
        self.buf
    }
}

// ------------------------------ documentation ------------------------------

/// Wrap words greedily to `width` columns; deterministic.
pub fn wrap(text: &str, width: usize) -> Vec<String> {
    let mut lines = Vec::new();
    let mut cur = String::new();
    for word in text.split_whitespace() {
        if cur.is_empty() {
            cur = word.to_string();
        } else if cur.len() + 1 + word.len() <= width {
            cur.push(' ');
            cur.push_str(word);
        } else {
            lines.push(std::mem::take(&mut cur));
            cur = word.to_string();
        }
    }
    if !cur.is_empty() {
        lines.push(cur);
    }
    lines
}

/// Shared module documentation as paragraphs of plain text (no comment
/// markers); the emitters add the language-specific framing.
pub fn doc_paragraphs(plan: &Plan, opts: &crate::transpile::EmitOptions) -> (String, Vec<String>) {
    let title = match &opts.title {
        Some(t) => t.clone(),
        None => default_title(plan),
    };
    let mut paragraphs: Vec<String> = Vec::new();
    paragraphs.push(
        "This program trains a regression model on train.csv, in which every row holds the \
         feature columns followed by the target as the last column, and then writes one \
         prediction per line for the rows of test_features.csv. All arithmetic is IEEE-754 \
         double precision in a fixed evaluation order, so repeated runs give identical bits."
            .to_string(),
    );
    match plan {
        Plan::Pipeline {
            target,
            stages,
            head,
        } => {
            if let Some(target) = target {
                paragraphs.push(format!(
                    "Before any feature stage, the training target is rewritten by a fitted \
                     target transform ({name}), which {doc} The feature stages and the head are \
                     trained on the transformed target, and every prediction is mapped back \
                     through the transform's inverse and then clamped to [-60, 60].",
                    name = target_fn_name(target.kind),
                    doc = target_doc(target)
                ));
            }
            let mut p = String::new();
            if stages.is_empty() {
                p.push_str("The model applies the regression head directly to the input columns.");
            } else {
                // State the append/replace and sanitize convention exactly once,
                // then list each stage compactly. Repeated identical stages refer
                // back to their first appearance instead of restating the same
                // sentence, so long pipelines carry no duplicated prose.
                p.push_str(&format!(
                    "The feature pipeline has {n} stage{plural}, applied in order; each stage is \
                     marked \"replaced\" when it overwrites the columns it receives or \"appended\" \
                     when it keeps them and adds its outputs alongside, and after every stage \
                     non-finite cells become 0.0 and each row is capped at 512 columns. ",
                    n = stages.len(),
                    plural = if stages.len() == 1 { "" } else { "s" }
                ));
                let mut seen_docs: Vec<(String, usize)> = Vec::new();
                for (i, stage) in stages.iter().enumerate() {
                    let n = i + 1;
                    let name = op_fn_name(&stage.op);
                    let mode = if stage.append { "appended" } else { "replaced" };
                    let doc = op_doc(&stage.op);
                    match seen_docs.iter().find(|(d, _)| *d == doc) {
                        Some((_, first)) => p.push_str(&format!(
                            "Stage {n} ({name}, {mode}) repeats the transform of stage {first}. "
                        )),
                        None => {
                            p.push_str(&format!("Stage {n} ({name}, {mode}) {doc} "));
                            seen_docs.push((doc, n));
                        }
                    }
                }
            }
            paragraphs.push(p.trim_end().to_string());
            paragraphs.push(format!(
                "The head ({name}) is {doc} Seeded stages draw weights from a permuted \
                 congruential generator (PCG XSH-RR 64/32; O'Neill, 2014) keyed by the stage seed \
                 and the input/output widths, so training is fully deterministic. Missing or \
                 malformed CSV cells parse as 0.0 and short rows are padded with zeros to the \
                 training width.",
                name = head_fn_name(head),
                doc = head_doc(head)
            ));
        }
        Plan::Symbolic { source, affine, .. } => {
            paragraphs.push(format!(
                "The model is the fixed closed-form expression f(row) = {source}, evaluated with \
                 guarded arithmetic: division by a magnitude under 1e-12 gives 0, exponents are \
                 clamped to [-8, 8], sqrt uses max(v, 0), log1p is 0 at or under -1, the logistic \
                 argument is clamped to [-30, 30], and the final value is clamped to [-1e12, 1e12] \
                 with non-finite results replaced by 0."
            ));
            if *affine {
                paragraphs.push(
                    "Predictions are a*f(row) + b where the slope a and intercept b are the \
                     ordinary least-squares line of the training target on f, a linear \
                     calibration of the closed form; outputs are clipped to \
                     [-60, 60]. Missing or malformed CSV cells parse as 0.0 and short rows are \
                     padded with zeros to the training width."
                        .to_string(),
                );
            } else {
                paragraphs.push(
                    "Predictions are f(row) directly (slope 1, intercept 0), clipped to \
                     [-60, 60]. Missing or malformed CSV cells parse as 0.0 and short rows are \
                     padded with zeros to the training width."
                        .to_string(),
                );
            }
        }
    }
    if !opts.feature_names.is_empty() {
        paragraphs.push(format!("Input columns: {}.", opts.feature_names.join(", ")));
    }
    if let Some(ctx) = &opts.context {
        paragraphs.push(ctx.clone());
    }
    (title, paragraphs)
}

fn default_title(plan: &Plan) -> String {
    match plan {
        Plan::Pipeline {
            target,
            stages,
            head,
        } => {
            let mut parts: Vec<&str> = Vec::new();
            if let Some(target) = target {
                parts.push(target_short(target.kind));
            }
            parts.extend(stages.iter().map(|s| op_short(&s.op)));
            parts.push(head_short(head));
            format!("Regression model: {}", parts.join(", then "))
        }
        Plan::Symbolic { affine, .. } => {
            if *affine {
                "Regression model: closed-form expression with least-squares calibration"
                    .to_string()
            } else {
                "Regression model: closed-form expression".to_string()
            }
        }
    }
}
