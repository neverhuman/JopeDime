use super::{MemberPlan, MemberPred};
use crate::backend::TrainSpec;
use crate::data::Matrix;
use crate::starforge_integration::{StarforgeConfig, StarforgeMemberPlan};
use crate::task::TaskKind;

pub(super) struct ResidualStarforgePred {
    pub oof: Vec<f32>,
    pub infer: Vec<f32>,
    pub plan: StarforgeMemberPlan,
}

#[cfg(feature = "starforge-cpu")]
#[allow(clippy::too_many_arguments)]
pub(super) fn append_starforge_members(
    members: &mut Vec<MemberPred>,
    member_plan_specs: &mut Vec<MemberPlan>,
    x: &Matrix,
    y: &[f32],
    xh: &Matrix,
    yh: &[f32],
    folds: &[(Vec<usize>, Vec<usize>)],
    reduced_cols: &[u32],
    top_cols: &[u32],
    ho_spec: &TrainSpec,
    task: TaskKind,
    n_classes: usize,
    cfg: &StarforgeConfig,
) {
    let _ = yh;
    if !cfg.wants_starforge() {
        return;
    }
    if !matches!(
        task,
        TaskKind::Binary | TaskKind::MultiLabel | TaskKind::Regression
    ) {
        eprintln!(
            "[starforge] skipping: supported Starforge tasks are binary, multilabel, and regression"
        );
        return;
    }
    let session = match crate::starforge_integration::load_session_for_task(cfg, task) {
        Ok(Some(session)) => session,
        Ok(None) => return,
        Err(err) if cfg.is_required() => panic!("Starforge setup failed: {err:#}"),
        Err(err) => {
            eprintln!("[starforge] setup failed; skipping Starforge candidates: {err:#}");
            return;
        }
    };
    let names = if task == TaskKind::Regression {
        ("starforge_regressor_reduced", "starforge_regressor_top500")
    } else {
        ("starforge_reduced", "starforge_top500")
    };
    for (name, cols) in [(names.0, reduced_cols), (names.1, top_cols)] {
        if cols.is_empty() {
            continue;
        }
        match session.candidate(name, x, y, xh, yh, folds, cols, cfg, task, n_classes) {
            Ok(candidate) => {
                eprintln!(
                    "[starforge] candidate={} features={} cv={:.5} lcb={:.5}",
                    candidate.name,
                    cols.len(),
                    candidate.mean_auc,
                    candidate.lcb_auc
                );
                member_plan_specs.push(MemberPlan {
                    name: candidate.name.clone(),
                    backend: "starforge".into(),
                    weight: 0.0,
                    spec: *ho_spec,
                    starforge: Some(candidate.plan),
                    math: None,
                });
                members.push(MemberPred {
                    name: candidate.name,
                    oof: candidate.oof,
                    infer: candidate.infer,
                    fold_auc: candidate.fold_auc,
                    mean_auc: candidate.mean_auc,
                    lcb_auc: candidate.lcb_auc,
                });
            }
            Err(err) if cfg.is_required() => panic!("Starforge candidate {name} failed: {err:#}"),
            Err(err) => eprintln!("[starforge] candidate {name} failed; skipping: {err:#}"),
        }
    }
}

#[cfg(feature = "starforge-cpu")]
#[allow(clippy::too_many_arguments)]
pub(super) fn residual_starforge_predictions(
    x: &Matrix,
    residual: &[f32],
    xh: &Matrix,
    folds: &[(Vec<usize>, Vec<usize>)],
    cols: &[u32],
    cfg: &StarforgeConfig,
) -> Option<ResidualStarforgePred> {
    if !cfg.wants_starforge() {
        return None;
    }
    let session =
        match crate::starforge_integration::load_session_for_task(cfg, TaskKind::Regression) {
            Ok(Some(session)) => session,
            Ok(None) => return None,
            Err(err) if cfg.is_required() => panic!("Starforge residual setup failed: {err:#}"),
            Err(err) => {
                eprintln!("[starforge] residual setup failed; skipping residual fusion: {err:#}");
                return None;
            }
        };
    let hold_dummy = vec![0.0; xh.n];
    match session.candidate(
        "starforge_residual_regressor",
        x,
        residual,
        xh,
        &hold_dummy,
        folds,
        cols,
        cfg,
        TaskKind::Regression,
        2,
    ) {
        Ok(candidate) => Some(ResidualStarforgePred {
            oof: candidate.oof,
            infer: candidate.infer,
            plan: candidate.plan,
        }),
        Err(err) if cfg.is_required() => panic!("Starforge residual candidate failed: {err:#}"),
        Err(err) => {
            eprintln!("[starforge] residual candidate failed; skipping residual fusion: {err:#}");
            None
        }
    }
}

#[cfg(not(feature = "starforge-cpu"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn residual_starforge_predictions(
    _x: &Matrix,
    _residual: &[f32],
    _xh: &Matrix,
    _folds: &[(Vec<usize>, Vec<usize>)],
    _cols: &[u32],
    cfg: &StarforgeConfig,
) -> Option<ResidualStarforgePred> {
    if cfg.wants_starforge() {
        eprintln!("[starforge] not compiled into this binary; skipping residual fusion");
    }
    None
}

#[cfg(not(feature = "starforge-cpu"))]
#[allow(clippy::too_many_arguments)]
pub(super) fn append_starforge_members(
    _members: &mut Vec<MemberPred>,
    _member_plan_specs: &mut Vec<MemberPlan>,
    _x: &Matrix,
    _y: &[f32],
    _xh: &Matrix,
    _yh: &[f32],
    _folds: &[(Vec<usize>, Vec<usize>)],
    _reduced_cols: &[u32],
    _top_cols: &[u32],
    _ho_spec: &TrainSpec,
    _task: TaskKind,
    _n_classes: usize,
    cfg: &StarforgeConfig,
) {
    if cfg.wants_starforge() {
        eprintln!("[starforge] not compiled into this binary; skipping Starforge candidates");
    }
}
