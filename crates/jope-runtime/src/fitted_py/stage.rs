use super::render::Py;
use feat_math::ops::FittedOp;
use feat_math::spec::{PlannedOp, PlannedStage};

/// The state the recipe's `*_apply` reads for one fitted stage (None for a stateless kind), after
/// checking it against the planned stage. Problems are pushed and the stage is skipped.
pub(super) fn stage_state(
    index: usize,
    planned: &PlannedStage,
    op: &FittedOp,
    problems: &mut Vec<String>,
) -> Option<Py> {
    use FittedOp as F;
    use PlannedOp as P;
    if fitted_append(op) != planned.append {
        problems.push(format!(
            "stage {index} append parameter differs from fitted state"
        ));
    }
    let mut differs = |name: &str| {
        problems.push(format!(
            "stage {index} {name} parameter differs from fitted state"
        ));
    };
    let same = |a: f64, b: f64| a.is_finite() && a.to_bits() == b.to_bits();
    let state = match (&planned.op, op) {
        (P::Standardize, F::Standardize { means, scales, .. }) => {
            Some(Py::Tuple(vec![Py::floats(means), Py::floats(scales)]))
        }
        (
            P::RobustScale,
            F::RobustScale {
                medians, scales, ..
            },
        ) => Some(Py::Tuple(vec![Py::floats(medians), Py::floats(scales)])),
        (P::Rank, F::Rank { sorted_cols, .. }) | (P::Blom, F::Blom { sorted_cols, .. }) => {
            Some(Py::rows(sorted_cols))
        }
        (
            P::QuantileBins { bins },
            F::QuantileBins {
                edges,
                bins: fitted,
                ..
            },
        )
        | (
            P::BinOnehot { bins },
            F::BinOnehot {
                edges,
                bins: fitted,
                ..
            },
        ) => {
            if bins != fitted {
                differs("bins");
            }
            Some(Py::rows(edges))
        }
        (P::Hinge { .. }, F::Hinge { knots, .. }) | (P::Spline { .. }, F::Spline { knots, .. }) => {
            Some(Py::rows(knots))
        }
        (P::WinsorClip { .. }, F::WinsorClip { lo, hi, .. }) => {
            Some(Py::Tuple(vec![Py::floats(lo), Py::floats(hi)]))
        }
        (P::Cheb { degree }, F::Cheb { degree: fitted, .. }) => {
            if degree != fitted {
                differs("degree");
            }
            None
        }
        (P::PolyInteract, F::PolyInteract { pairs, .. })
        | (P::PairProd { .. }, F::PairProd { pairs, .. }) => Some(Py::pairs(pairs)),
        (P::RffCos { .. }, F::RffCos { w, b, .. }) | (P::RffSin { .. }, F::RffSin { w, b, .. }) => {
            Some(Py::Tuple(vec![Py::rows(w), Py::floats(b)]))
        }
        (P::RandProj { .. }, F::RandProj { w, .. }) => Some(Py::rows(w)),
        (
            P::LandmarkDist { gamma, .. },
            F::LandmarkDist {
                landmarks,
                gamma: fitted,
                ..
            },
        )
        | (
            P::Nystrom { gamma, .. },
            F::Nystrom {
                landmarks,
                gamma: fitted,
                ..
            },
        ) => {
            if !same(*fitted, *gamma) {
                differs("gamma");
            }
            Some(Py::rows(landmarks))
        }
        (P::ObliqueHinge { .. }, F::ObliqueHinge { w, t, .. }) => {
            Some(Py::Tuple(vec![Py::rows(w), Py::floats(t)]))
        }
        (P::TargetBinEncode { .. }, F::TargetBinEncode { edges, enc, .. }) => {
            Some(Py::Tuple(vec![Py::rows(edges), Py::rows(enc)]))
        }
        (P::Step { t }, F::Step { t: fitted, .. }) => {
            if !same(*fitted, *t) {
                differs("t");
            }
            None
        }
        (P::Symlog { c }, F::Symlog { c: fitted, .. }) => {
            if !same(*fitted, *c) {
                differs("c");
            }
            None
        }
        (P::Lag { k }, F::Lag { k: fitted, .. }) | (P::Roll { k }, F::Roll { k: fitted, .. }) => {
            if k != fitted {
                differs("k");
            }
            None
        }
        // The recipe reads the fit-time width-capped k from the stage state.
        (P::FftMag { .. }, F::FftMag { k, .. }) => Some(Py::I(*k)),
        (
            P::SvdProject { .. },
            F::SvdProject {
                means, components, ..
            },
        ) => Some(Py::Tuple(vec![Py::floats(means), Py::rows(components)])),
        (P::SelectCorr { .. }, F::SelectCorr { indices, .. }) => Some(Py::ints(indices)),
        (
            P::Impute { .. },
            F::Impute {
                fill,
                indicator_cols,
                ..
            },
        ) => Some(Py::Tuple(vec![Py::floats(fill), Py::ints(indicator_cols)])),
        (P::PairRatio { .. }, F::PairRatio { pairs, eps, .. }) => {
            Some(Py::Tuple(vec![Py::pairs(pairs), Py::floats(eps)]))
        }
        (P::MapExpr { source, .. }, F::MapExpr { source: fitted, .. })
        | (P::DeriveExpr { source, .. }, F::DeriveExpr { source: fitted, .. }) => {
            if source != fitted {
                differs("expr");
            }
            None
        }
        (P::RowStats, F::RowStats { .. })
        | (P::Sigmoid, F::Sigmoid { .. })
        | (P::NormRow, F::NormRow { .. })
        | (P::Polar, F::Polar { .. })
        | (P::RowSpectral, F::RowSpectral { .. })
        | (P::RowDiffStats, F::RowDiffStats { .. }) => None,
        (planned, _) => {
            problems.push(format!(
                "fitted {} where the plan has {}",
                fitted_kind(op),
                planned_kind(planned)
            ));
            None
        }
    };
    state
}

fn fitted_append(op: &FittedOp) -> bool {
    use FittedOp as F;
    match op {
        F::Standardize { append, .. }
        | F::RobustScale { append, .. }
        | F::Rank { append, .. }
        | F::Blom { append, .. }
        | F::QuantileBins { append, .. }
        | F::BinOnehot { append, .. }
        | F::Hinge { append, .. }
        | F::WinsorClip { append, .. }
        | F::Cheb { append, .. }
        | F::PolyInteract { append, .. }
        | F::RffCos { append, .. }
        | F::RffSin { append, .. }
        | F::RandProj { append, .. }
        | F::LandmarkDist { append, .. }
        | F::ObliqueHinge { append, .. }
        | F::RowStats { append }
        | F::PairProd { append, .. }
        | F::TargetBinEncode { append, .. }
        | F::Sigmoid { append }
        | F::Step { append, .. }
        | F::NormRow { append }
        | F::Polar { append }
        | F::Lag { append, .. }
        | F::Roll { append, .. }
        | F::FftMag { append, .. }
        | F::RowSpectral { append }
        | F::RowDiffStats { append }
        | F::SvdProject { append, .. }
        | F::Nystrom { append, .. }
        | F::MapExpr { append, .. }
        | F::DeriveExpr { append, .. }
        | F::Symlog { append, .. }
        | F::SelectCorr { append, .. }
        | F::Impute { append, .. }
        | F::PairRatio { append, .. }
        | F::Spline { append, .. } => *append,
    }
}

/// The program name of a fitted stage, for refusal messages.
fn fitted_kind(op: &FittedOp) -> &'static str {
    use FittedOp as F;
    match op {
        F::Standardize { .. } => "standardize",
        F::RobustScale { .. } => "robust_scale",
        F::Rank { .. } => "rank",
        F::Blom { .. } => "blom",
        F::QuantileBins { .. } => "quantile_bins",
        F::BinOnehot { .. } => "bin_onehot",
        F::Hinge { .. } => "hinge",
        F::WinsorClip { .. } => "winsor_clip",
        F::Cheb { .. } => "cheb",
        F::PolyInteract { .. } => "poly_interact",
        F::RffCos { .. } => "rff_cos",
        F::RffSin { .. } => "rff_sin",
        F::RandProj { .. } => "rand_proj",
        F::LandmarkDist { .. } => "landmark_dist",
        F::ObliqueHinge { .. } => "oblique_hinge",
        F::RowStats { .. } => "row_stats",
        F::PairProd { .. } => "pair_prod",
        F::TargetBinEncode { .. } => "target_bin_encode",
        F::Sigmoid { .. } => "sigmoid",
        F::Step { .. } => "step",
        F::NormRow { .. } => "norm_row",
        F::Polar { .. } => "polar",
        F::Lag { .. } => "lag",
        F::Roll { .. } => "roll",
        F::FftMag { .. } => "fft_mag",
        F::RowSpectral { .. } => "row_spectral",
        F::RowDiffStats { .. } => "row_diff_stats",
        F::SvdProject { .. } => "svd_project",
        F::Nystrom { .. } => "nystrom",
        F::MapExpr { .. } => "map",
        F::DeriveExpr { .. } => "derive",
        F::Symlog { .. } => "symlog",
        F::SelectCorr { .. } => "select_corr",
        F::Impute { .. } => "impute",
        F::PairRatio { .. } => "pair_ratio",
        F::Spline { .. } => "spline",
    }
}

/// The program name of a planned stage, for refusal messages.
fn planned_kind(op: &PlannedOp) -> &'static str {
    use PlannedOp as P;
    match op {
        P::Standardize => "standardize",
        P::RobustScale => "robust_scale",
        P::Rank => "rank",
        P::Blom => "blom",
        P::QuantileBins { .. } => "quantile_bins",
        P::BinOnehot { .. } => "bin_onehot",
        P::Hinge { .. } => "hinge",
        P::WinsorClip { .. } => "winsor_clip",
        P::Cheb { .. } => "cheb",
        P::PolyInteract => "poly_interact",
        P::RffCos { .. } => "rff_cos",
        P::RffSin { .. } => "rff_sin",
        P::RandProj { .. } => "rand_proj",
        P::LandmarkDist { .. } => "landmark_dist",
        P::ObliqueHinge { .. } => "oblique_hinge",
        P::RowStats => "row_stats",
        P::PairProd { .. } => "pair_prod",
        P::TargetBinEncode { .. } => "target_bin_encode",
        P::Sigmoid => "sigmoid",
        P::Step { .. } => "step",
        P::NormRow => "norm_row",
        P::Polar => "polar",
        P::Lag { .. } => "lag",
        P::Roll { .. } => "roll",
        P::FftMag { .. } => "fft_mag",
        P::RowSpectral => "row_spectral",
        P::RowDiffStats => "row_diff_stats",
        P::SvdProject { .. } => "svd_project",
        P::Nystrom { .. } => "nystrom",
        P::MapExpr { .. } => "map",
        P::DeriveExpr { .. } => "derive",
        P::Symlog { .. } => "symlog",
        P::SelectCorr { .. } => "select_corr",
        P::Impute { .. } => "impute",
        P::PairRatio { .. } => "pair_ratio",
        P::Spline { .. } => "spline",
        P::RowVectorV2 { .. } => "a row-vector v2 stage",
        P::RowVectorInputV2 { .. } => "a row-vector v2 input",
        P::ScalarV2 { .. } => "a scalar v2 stage",
    }
}
