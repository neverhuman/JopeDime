use super::types::{EnsembleResult, FuserMode, FuserPlan, MathMemberPlan, MemberPlan};
use crate::backend::{Backend, TrainSpec};
use crate::data::Matrix;
use crate::progress::{Event, Sink};
use crate::starforge_integration::StarforgeConfig;
use crate::task::TaskKind;
use std::time::Instant;

mod binary;
mod regression;

/// Build a fold-pure OOF library over the champion features, then pick the best
/// fuser by task-aware OOF lower-confidence-bound and report its blind holdout.
#[allow(clippy::too_many_arguments)]
pub fn select_model(
    cat: &dyn Backend,
    xgb: &dyn Backend,
    lgb: &dyn Backend,
    x: &Matrix,
    y: &[f32],
    xh: &Matrix,
    yh: &[f32],
    folds: &[(Vec<usize>, Vec<usize>)],
    cols: &[u32],
    cv_spec: &TrainSpec,
    ho_spec: &TrainSpec,
    task: TaskKind,
    n_classes: usize,
    starforge_cfg: &StarforgeConfig,
    starforge_top_cols: &[u32],
    invented: Option<&crate::invention::InventionReport>,
    sink: &dyn Sink,
    t0: Instant,
) -> EnsembleResult {
    let phase = 5u8;
    let mut e = Event::new(phase, "start", t0.elapsed().as_secs_f64());
    e.final_features = Some(cols.len() as u64);
    sink.emit(&e);

    let native = if matches!(task, TaskKind::Binary | TaskKind::MultiLabel) {
        binary::select_binary_or_multilabel(
            cat,
            xgb,
            lgb,
            x,
            y,
            xh,
            yh,
            folds,
            cols,
            cv_spec,
            ho_spec,
            task,
            n_classes,
            starforge_cfg,
            starforge_top_cols,
            sink,
            t0,
            phase,
        )
    } else {
        regression::select_regression_or_multiclass(
            cat,
            x,
            y,
            xh,
            yh,
            folds,
            cols,
            cv_spec,
            ho_spec,
            task,
            n_classes,
            starforge_cfg,
            starforge_top_cols,
            sink,
            t0,
            phase,
        )
    };
    let selected = select_invented_if_better(native, invented, ho_spec);
    emit_final_done(&selected, sink, t0, phase, task);
    selected
}

fn select_invented_if_better(
    native: EnsembleResult,
    invented: Option<&crate::invention::InventionReport>,
    ho_spec: &TrainSpec,
) -> EnsembleResult {
    let Some(best) = invented.and_then(|report| {
        report.finalists.iter().max_by(|a, b| {
            a.cv_mean
                .total_cmp(&b.cv_mean)
                .then_with(|| b.name.cmp(&a.name))
        })
    }) else {
        return native;
    };
    if best.cv_mean > native.final_cv_auc {
        invented_result(best, native, ho_spec)
    } else {
        native
    }
}

fn invented_result(
    best: &crate::invention::InventedFinalist,
    native: EnsembleResult,
    ho_spec: &TrainSpec,
) -> EnsembleResult {
    let plan = FuserPlan {
        name: best.name.clone(),
        mode: FuserMode::WeightedBlend,
        rank_outputs: false,
        members: vec![MemberPlan {
            name: best.name.clone(),
            backend: "math".into(),
            weight: 1.0,
            spec: *ho_spec,
            starforge: None,
            math: Some(MathMemberPlan {
                dsl: best.dsl.clone(),
            }),
        }],
        residual: None,
    };
    EnsembleResult {
        final_model: best.name.clone(),
        final_cv_auc: best.cv_mean,
        final_holdout_auc: best.holdout,
        final_holdout_preds: best.holdout_preds.clone(),
        catboost_cv: native.catboost_cv,
        xgb_cv: native.xgb_cv,
        lgb_cv: native.lgb_cv,
        ensemble_cv: native.ensemble_cv,
        oof_lcb: best.cv_mean,
        active_members: 1,
        selected_fuser: best.name.clone(),
        family: native.family,
        single_lcb: native.single_lcb,
        multi_lcb: native.multi_lcb,
        gate_passed: native.gate_passed,
        sibling_count: native.sibling_count,
        final_plan: Some(plan),
    }
}

fn emit_final_done(
    selected: &EnsembleResult,
    sink: &dyn Sink,
    t0: Instant,
    phase: u8,
    task: TaskKind,
) {
    let mut e = Event::new(phase, "done", t0.elapsed().as_secs_f64());
    e.model = Some(selected.final_model.clone());
    e.metric = Some(crate::metrics::metric_name(task).to_string());
    e.best_val_auc = Some(selected.final_cv_auc);
    e.best_holdout_auc = Some(selected.final_holdout_auc);
    sink.emit(&e);
}
