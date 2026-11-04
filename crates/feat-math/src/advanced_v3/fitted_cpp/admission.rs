/// Exhaustive admission check. Every fitted enum is matched explicitly so a
/// newly added registry variant fails compilation until its C++ semantics are
/// deliberately classified.
pub fn supports_fitted_cpp17(model: &BoundModel) -> Result<()> {
    model.validate_binding()?;
    match &model.schema.row_semantics {
        RowSemantics::Iid | RowSemantics::Ordered { .. } => {}
    }
    match &model.predictor {
        BoundPredictor::Pipeline {
            target,
            ops,
            head,
            calibration,
            ..
        } => {
            check_target(target);
            check_calibration(calibration);
            for op in ops {
                check_op(op)?;
            }
            check_head(head)
        }
        BoundPredictor::Legacy(crate::model::Model::Pipeline { target, ops, head }) => {
            check_target(target);
            for op in ops {
                check_op(&V3Op::Legacy(op.clone()))?;
            }
            check_head(&V3Head::Legacy((**head).clone()))
        }
        BoundPredictor::Legacy(crate::model::Model::Symbolic { expr, .. }) => {
            expr.to_cpp17(false).map(|_| ())
        }
        BoundPredictor::Legacy(crate::model::Model::Portfolio(model)) => check_portfolio(model),
    }
}

fn check_target(target: &Option<FittedTarget>) {
    match target {
        None
        | Some(FittedTarget::RankGauss { .. })
        | Some(FittedTarget::Log1p)
        | Some(FittedTarget::Winsor { .. })
        | Some(FittedTarget::ZScore { .. }) => {}
    }
}

fn check_calibration(calibration: &CalibrationModel) {
    match calibration {
        CalibrationModel::None
        | CalibrationModel::Platt { .. }
        | CalibrationModel::Isotonic { .. } => {}
    }
}

fn check_op(op: &V3Op) -> Result<()> {
    match op {
        V3Op::Legacy(op) => match op {
            FittedOp::Standardize { .. }
            | FittedOp::RobustScale { .. }
            | FittedOp::Rank { .. }
            | FittedOp::Blom { .. }
            | FittedOp::QuantileBins { .. }
            | FittedOp::BinOnehot { .. }
            | FittedOp::Hinge { .. }
            | FittedOp::WinsorClip { .. }
            | FittedOp::Cheb { .. }
            | FittedOp::PolyInteract { .. }
            | FittedOp::RffCos { .. }
            | FittedOp::RffSin { .. }
            | FittedOp::RandProj { .. }
            | FittedOp::LandmarkDist { .. }
            | FittedOp::ObliqueHinge { .. }
            | FittedOp::PairProd { .. }
            | FittedOp::TargetBinEncode { .. }
            | FittedOp::Sigmoid { .. }
            | FittedOp::Step { .. }
            | FittedOp::NormRow { .. }
            | FittedOp::Polar { .. }
            | FittedOp::Lag { .. }
            | FittedOp::Roll { .. }
            | FittedOp::SvdProject { .. }
            | FittedOp::Nystrom { .. }
            | FittedOp::Symlog { .. }
            | FittedOp::SelectCorr { .. }
            | FittedOp::Impute { .. }
            | FittedOp::PairRatio { .. }
            | FittedOp::Spline { .. } => Ok(()),
            FittedOp::RowStats { .. }
            | FittedOp::FftMag { .. }
            | FittedOp::RowSpectral { .. }
            | FittedOp::RowDiffStats { .. }
            | FittedOp::MapExpr { .. }
            | FittedOp::DeriveExpr { .. } => Ok(()),
        },
        V3Op::MaxAbs { .. }
        | V3Op::YeoJohnson { .. }
        | V3Op::PiecewiseLinear { .. }
        | V3Op::CorrScreen { .. }
        | V3Op::PlsProject { .. } => Ok(()),
        V3Op::Lag { .. } | V3Op::Roll { .. } => Ok(()),
    }
}

fn check_head(head: &V3Head) -> Result<()> {
    match head {
        V3Head::Legacy(head) => match head {
            FittedHead::Ridge(_)
            | FittedHead::RidgeGcv(_)
            | FittedHead::ElasticNet(_)
            | FittedHead::Gbm(_)
            | FittedHead::Ensemble { .. }
            | FittedHead::HistGbm2(_)
            | FittedHead::Krr(_)
            | FittedHead::Gpr(_)
            | FittedHead::Ksmooth(_)
            | FittedHead::Hurdle2(_)
            | FittedHead::Stack(_) => Ok(()),
            FittedHead::GprKernel(_) => Ok(()),
            FittedHead::Portfolio(model) => check_portfolio(model),
        },
        V3Head::LogitRidge(_)
        | V3Head::TabBoost(_)
        | V3Head::ExtraTrees(_)
        | V3Head::GamBoost(_)
        | V3Head::Pls(_) => Ok(()),
        V3Head::ClassificationPortfolio(model) => {
            for member in &model.members {
                check_head(member)?;
            }
            Ok(())
        }
        V3Head::MulticlassRidge(_)
        | V3Head::MulticlassOvrLogistic(_)
        | V3Head::MulticlassDiagonalLda(_)
        | V3Head::MulticlassNearestCentroid(_) => {
            bail!("multiclass probability matrices are not supported by the scalar C++ ABI")
        }
    }
}

fn check_legacy_model(model: &crate::model::Model) -> Result<()> {
    match model {
        crate::model::Model::Pipeline { target, ops, head } => {
            check_target(target);
            for op in ops {
                check_op(&V3Op::Legacy(op.clone()))?;
            }
            check_head(&V3Head::Legacy((**head).clone()))
        }
        crate::model::Model::Symbolic { expr, .. } => expr.to_cpp17(false).map(|_| ()),
        crate::model::Model::Portfolio(model) => check_portfolio(model),
    }
}

fn check_portfolio(model: &crate::portfolio::PortfolioModel) -> Result<()> {
    if let Some(projection) = &model.proj {
        check_op(&V3Op::Legacy(projection.clone()))?;
    }
    for member in &model.members {
        match &member.state {
            FittedMemberState::Fitted(model) => check_legacy_model(model)?,
            FittedMemberState::GroupMean(_)
            | FittedMemberState::Mean
            | FittedMemberState::Skipped => {}
        }
    }
    Ok(())
}
