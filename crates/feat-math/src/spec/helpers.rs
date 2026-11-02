use super::*;

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
        // B05–B07: scalar-v2 named registration.
        PlannedOp::ScalarV2 { .. } => &[],
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
                    PlannedOp::ScalarV2 { program, .. } => {
                        add_helper(&mut set, Helper::Clamp);
                        expr_helpers(
                            &crate::kernel::scalar_program::emit::old_helper_features(program)
                                .expect("admitted scalar program"),
                            &mut set,
                        );
                    }
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
