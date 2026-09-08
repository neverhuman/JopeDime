//! Final-stage fusion. On the champion feature set we build a fold-pure OUT-OF-FOLD prediction library
//! (every train row predicted by a model that did NOT train on it), then choose the best of
//!   { best single, soft-vote, bagged rank-aware Caruana greedy ensemble selection (GES) }
//! by cross-validated lower-confidence-bound AUC, and report that fuser's blind holdout. The holdout is
//! NEVER used for selection. This supersedes the old "CatBoost vs soft-vote by CV" picker while keeping
//! the same entry point + `EnsembleResult` fields.

mod ges;
mod oof;
mod selection;
mod starforge;
mod types;

pub use ges::{bagged_ges, GesConfig};
pub use oof::build_oof_member;
pub use selection::select_model;
pub use types::{
    EnsembleResult, FuserMode, FuserPlan, MathMemberPlan, MemberPlan, MemberPred, ResidualFuserPlan,
};

pub(crate) use ges::{blend, rank01_per_fold};

/// Lower-confidence-bound t used for fusion selection (matches the EDA selector's lcb_t).
pub(crate) const LCB_T: f64 = 1.0;
