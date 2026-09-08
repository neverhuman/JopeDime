use crate::backend::TrainSpec;
use crate::starforge_integration::StarforgeMemberPlan;
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MathMemberPlan {
    pub dsl: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MemberPlan {
    pub name: String,
    pub backend: String,
    pub weight: f64,
    pub spec: TrainSpec,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub starforge: Option<StarforgeMemberPlan>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub math: Option<MathMemberPlan>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum FuserMode {
    #[default]
    WeightedBlend,
    Residual,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct ResidualFuserPlan {
    pub base_member: String,
    pub residual_member: String,
    pub alpha: f64,
    pub clamp_min: f32,
    pub clamp_max: f32,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct FuserPlan {
    pub name: String,
    #[serde(default)]
    pub mode: FuserMode,
    pub rank_outputs: bool,
    pub members: Vec<MemberPlan>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub residual: Option<ResidualFuserPlan>,
}

pub struct EnsembleResult {
    pub final_model: String,
    pub final_cv_auc: f64,
    pub final_holdout_auc: f64,
    /// Per-row holdout probabilities of the CV/OOF-selected final fuser (row order = hold CSV).
    /// Used to dump jain's predictions for an apples-to-apples bootstrap-CI / pairwise comparison.
    pub final_holdout_preds: Vec<f32>,
    pub catboost_cv: f64,
    pub xgb_cv: f64,
    pub lgb_cv: f64,
    pub ensemble_cv: f64,
    // --- OOF-library + GES additions ---
    pub oof_lcb: f64,
    pub active_members: usize,
    pub selected_fuser: String,
    // --- multi-target stacking additions (defaults describe the single-target path) ---
    pub family: String,    // "single-target" | "multi-target"
    pub single_lcb: f64,   // single-target ceiling OOF-LCB
    pub multi_lcb: f64,    // best multi-target/stacked OOF-LCB (NaN if no siblings)
    pub gate_passed: bool, // did the multi-target path beat the single ceiling?
    pub sibling_count: usize,
    /// Refit/persistence plan for serving. `None` means the selected path contains a pure-Rust
    /// member/fuser that is not yet serializable as a native artifact.
    pub final_plan: Option<FuserPlan>,
}

/// One candidate member's fold-pure OOF predictions (train) + full-train inference predictions (holdout).
pub struct MemberPred {
    pub name: String,
    pub oof: Vec<f32>,
    pub infer: Vec<f32>,
    pub fold_auc: Vec<f64>,
    pub mean_auc: f64,
    pub lcb_auc: f64,
}
