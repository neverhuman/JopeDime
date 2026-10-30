//! Objective rules applied while constructing a run plan.
//!
//! The head kinds here are plan-facing names. They intentionally do not depend
//! on `feat-math`; the plan owner converts a selected implementation head to a
//! `HeadKind` before disclosing its fitting behavior.

/// Plan-facing head kinds mirrored from `feat_math::heads::FittedHead`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum HeadKind {
    Ridge,
    RidgeGcv,
    ElasticNet,
    Gbm,
    Ensemble,
    Krr,
    Gpr,
    GprKernel,
    Ksmooth,
    Stack,
    Portfolio,
}

impl HeadKind {
    /// Parse a plan head name or its supported operator alias.
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "Ridge" | "ridge" | "logistic" | "ridge_regression" => Some(Self::Ridge),
            "RidgeGcv" | "ridge_gcv" | "gcv_ridge_regression" => Some(Self::RidgeGcv),
            "ElasticNet" | "elasticnet" | "elastic_net" => Some(Self::ElasticNet),
            "Gbm" | "gbm" | "histgbm" | "gradient_boosted_trees" => Some(Self::Gbm),
            "Ensemble" | "ensemble" | "ensemble_ridge_gbm" | "blended_regression" => {
                Some(Self::Ensemble)
            }
            "Krr" | "krr" | "kernel_ridge_regression" => Some(Self::Krr),
            "Gpr" | "gpr" | "gaussian_process_ard_regression" => Some(Self::Gpr),
            "GprKernel" | "gpr_kernel" | "gaussian_process_composite_kernel_regression" => {
                Some(Self::GprKernel)
            }
            "Ksmooth" | "ksmooth" | "nadaraya_watson_regression" => Some(Self::Ksmooth),
            "Stack" | "stack" | "greedy_ensemble_selection" => Some(Self::Stack),
            "Portfolio" | "portfolio" | "internally_validated_ensemble" => Some(Self::Portfolio),
            _ => None,
        }
    }
}

/// How a selected head uses the user-provided objective.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Fitting {
    Native,
    SelectionOnly,
}

/// Public error code returned when an order-dependent objective lacks order.
pub const OBJECTIVE_REQUIRES_ORDER: &str = "OBJECTIVE_REQUIRES_ORDER";

/// Reject an order-dependent objective when the run has no time column.
///
/// `objective_name` is the normalized objective name, optionally followed by
/// the runtime grammar's `:parameter=value` suffix.
pub fn validate_order(objective_name: &str, time_column: Option<&str>) -> Result<(), &'static str> {
    let base_name = objective_name.split(':').next().unwrap_or_default();
    let requires_order = matches!(base_name, "returns" | "drawdown" | "mase");

    if requires_order && time_column.is_none() {
        Err(OBJECTIVE_REQUIRES_ORDER)
    } else {
        Ok(())
    }
}

/// All current heads fit their own fixed loss; the objective only selects them.
pub fn fitting_for(head: HeadKind) -> Fitting {
    match head {
        HeadKind::Ridge => Fitting::SelectionOnly,
        HeadKind::RidgeGcv => Fitting::SelectionOnly,
        HeadKind::ElasticNet => Fitting::SelectionOnly,
        HeadKind::Gbm => Fitting::SelectionOnly,
        HeadKind::Ensemble => Fitting::SelectionOnly,
        HeadKind::Krr => Fitting::SelectionOnly,
        HeadKind::Gpr => Fitting::SelectionOnly,
        HeadKind::GprKernel => Fitting::SelectionOnly,
        HeadKind::Ksmooth => Fitting::SelectionOnly,
        HeadKind::Stack => Fitting::SelectionOnly,
        HeadKind::Portfolio => Fitting::SelectionOnly,
    }
}

#[cfg(test)]
mod tests {
    use super::{fitting_for, validate_order, Fitting, HeadKind, OBJECTIVE_REQUIRES_ORDER};

    const HEADS: [HeadKind; 11] = [
        HeadKind::Ridge,
        HeadKind::RidgeGcv,
        HeadKind::ElasticNet,
        HeadKind::Gbm,
        HeadKind::Ensemble,
        HeadKind::Krr,
        HeadKind::Gpr,
        HeadKind::GprKernel,
        HeadKind::Ksmooth,
        HeadKind::Stack,
        HeadKind::Portfolio,
    ];

    #[test]
    fn each_order_dependent_objective_requires_time_column() {
        for objective in ["returns", "drawdown", "mase"] {
            assert_eq!(
                validate_order(objective, None),
                Err(OBJECTIVE_REQUIRES_ORDER),
                "{objective} without time_column must be refused"
            );
            assert_eq!(
                validate_order(objective, Some("event_time")),
                Ok(()),
                "{objective} with time_column is ordered"
            );
        }
    }

    #[test]
    fn objective_parameters_do_not_hide_order_requirement() {
        assert_eq!(
            validate_order("returns:window=5", None),
            Err(OBJECTIVE_REQUIRES_ORDER)
        );
    }

    #[test]
    fn ordinary_objectives_do_not_require_time_column() {
        assert_eq!(validate_order("rmse", None), Ok(()));
    }

    #[test]
    fn every_plan_head_name_maps_to_its_kind() {
        let cases = [
            ("ridge", HeadKind::Ridge),
            ("ridge_gcv", HeadKind::RidgeGcv),
            ("elasticnet", HeadKind::ElasticNet),
            ("gbm", HeadKind::Gbm),
            ("ensemble", HeadKind::Ensemble),
            ("krr", HeadKind::Krr),
            ("gpr", HeadKind::Gpr),
            ("gpr_kernel", HeadKind::GprKernel),
            ("ksmooth", HeadKind::Ksmooth),
            ("stack", HeadKind::Stack),
            ("portfolio", HeadKind::Portfolio),
        ];
        for (name, expected) in cases {
            assert_eq!(HeadKind::from_name(name), Some(expected), "{name}");
        }
        assert_eq!(HeadKind::from_name("unknown"), None);
    }

    #[test]
    fn every_head_discloses_selection_only_fitting() {
        for head in HEADS {
            assert_eq!(fitting_for(head), Fitting::SelectionOnly, "{head:?}");
        }
    }
}
