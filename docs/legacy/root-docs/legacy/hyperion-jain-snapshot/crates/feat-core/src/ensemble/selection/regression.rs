use super::super::oof::build_oof_member;
use super::super::starforge::append_starforge_members;
use super::super::{blend, EnsembleResult, FuserPlan, MemberPlan, MemberPred, LCB_T};
use crate::backend::{Backend, TrainSpec};
use crate::data::Matrix;
use crate::metrics::task_score;
use crate::progress::{Event, Sink};
use crate::starforge_integration::StarforgeConfig;
use crate::task::TaskKind;
use std::time::Instant;

/// Task-aware per-fold score + lower-confidence-bound for an OOF prediction vector.
fn fold_score_lcb(
    pred: &[f32],
    y: &[f32],
    folds: &[(Vec<usize>, Vec<usize>)],
    task: TaskKind,
    n_classes: usize,
    t: f64,
) -> (f64, f64) {
    let scores: Vec<f64> = folds
        .iter()
        .map(|(_, va)| {
            let yv: Vec<f32> = va.iter().map(|&i| y[i]).collect();
            let pv: Vec<f32> = va.iter().map(|&i| pred[i]).collect();
            task_score(task, &yv, &pv, n_classes)
        })
        .collect();
    let k = scores.len().max(1) as f64;
    let mean = scores.iter().sum::<f64>() / k;
    let var = scores.iter().map(|s| (s - mean) * (s - mean)).sum::<f64>() / k;
    (mean, mean - t * var.sqrt() / k.sqrt())
}

#[allow(clippy::too_many_arguments)]
pub(super) fn select_regression_or_multiclass(
    cat: &dyn Backend,
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
    sink: &dyn Sink,
    t0: Instant,
    phase: u8,
) -> EnsembleResult {
    let metric = crate::metrics::metric_name(task);
    let emit = |name: &str, cvv: f64, holdout: Option<f64>| {
        let mut e = Event::new(phase, "model", t0.elapsed().as_secs_f64());
        e.model = Some(name.into());
        e.metric = Some(metric.to_string());
        e.best_val_auc = Some(cvv);
        e.best_holdout_auc = holdout;
        sink.emit(&e);
    };
    // Same as `emit`, but carries the member's per-fold CV scores for the cockpit CV heatmap.
    let emit_member = |mp: &MemberPred| {
        let mut e = Event::new(phase, "model", t0.elapsed().as_secs_f64());
        e.model = Some(mp.name.clone());
        e.metric = Some(metric.to_string());
        e.best_val_auc = Some(mp.mean_auc);
        e.best_holdout_auc = Some(task_score(task, yh, &mp.infer, n_classes));
        if !mp.fold_auc.is_empty() {
            e.folds = Some(mp.fold_auc.clone());
        }
        sink.emit(&e);
    };

    let cat_m = build_oof_member(
        "catboost", cat, x, y, xh, folds, cols, cv_spec, ho_spec, task, n_classes,
    );
    emit_member(&cat_m);

    const BMA_MAX_N: usize = 12_000;
    let catboost_cv = cat_m.mean_auc;
    let mut members = vec![cat_m];
    let mut plans: Vec<Option<MemberPlan>> = vec![Some(MemberPlan {
        name: "catboost".into(),
        backend: "catboost".into(),
        weight: 1.0,
        spec: *ho_spec,
        starforge: None,
        math: None,
    })];
    if task == TaskKind::Regression && x.n <= BMA_MAX_N {
        let bma = crate::learners::bma::BmaBackend::gaussian();
        let bma_m = build_oof_member(
            "bma", &bma, x, y, xh, folds, cols, cv_spec, ho_spec, task, n_classes,
        );
        emit_member(&bma_m);
        let bma_idx = members.len();
        members.push(bma_m);
        plans.push(None);

        let n = y.len();
        let oof_cols = [members[0].oof.clone(), members[bma_idx].oof.clone()];
        let mut gram = [0f64; 4];
        let mut aty = [0f64; 2];
        for a in 0..2 {
            for i in 0..n {
                aty[a] += oof_cols[a][i] as f64 * y[i] as f64;
            }
            for b in 0..2 {
                gram[a * 2 + b] = (0..n)
                    .map(|i| oof_cols[a][i] as f64 * oof_cols[b][i] as f64)
                    .sum();
            }
        }
        let w = crate::linalg::nnls_cd(&gram, &aty, 2, 300);
        if w.iter().sum::<f64>() > 1e-9 {
            let blend_oof = blend(&oof_cols, &w);
            let (blend_mean, blend_lcb) =
                fold_score_lcb(&blend_oof, y, folds, task, n_classes, LCB_T);
            let blend_infer = blend(
                &[members[0].infer.clone(), members[bma_idx].infer.clone()],
                &w,
            );
            emit(
                "bma_blend",
                blend_mean,
                Some(task_score(task, yh, &blend_infer, n_classes)),
            );
            members.push(MemberPred {
                name: "bma_blend".into(),
                oof: blend_oof,
                infer: blend_infer,
                fold_auc: vec![],
                mean_auc: blend_mean,
                lcb_auc: blend_lcb,
            });
            plans.push(None);
        }
        if std::env::var("JAIN_DEBUG_BMA").is_ok() {
            for member in &members {
                let hold = task_score(task, yh, &member.infer, n_classes);
                eprintln!(
                    "[bma-dbg] {} lcb={:.4} oof={:.4} HOLD={hold:.4}",
                    member.name, member.lcb_auc, member.mean_auc
                );
            }
        }
    }

    if task == TaskKind::Regression {
        let mut sf_members = Vec::new();
        let mut sf_plans = Vec::new();
        append_starforge_members(
            &mut sf_members,
            &mut sf_plans,
            x,
            y,
            xh,
            yh,
            folds,
            cols,
            starforge_top_cols,
            ho_spec,
            task,
            n_classes,
            starforge_cfg,
        );
        for (member, plan) in sf_members.into_iter().zip(sf_plans) {
            emit_member(&member);
            members.push(member);
            plans.push(Some(MemberPlan {
                weight: 1.0,
                math: None,
                ..plan
            }));
        }
    }

    let persistable: Vec<usize> = plans
        .iter()
        .enumerate()
        .filter_map(|(i, plan)| plan.as_ref().map(|_| i))
        .collect();
    let mut blend_plan = None;
    if task == TaskKind::Regression && persistable.len() >= 2 {
        let oof_cols: Vec<Vec<f32>> = persistable
            .iter()
            .map(|&i| members[i].oof.clone())
            .collect();
        let mut gram = vec![0.0; persistable.len() * persistable.len()];
        let mut aty = vec![0.0; persistable.len()];
        for a in 0..persistable.len() {
            for i in 0..y.len() {
                aty[a] += oof_cols[a][i] as f64 * y[i] as f64;
            }
            for b in 0..persistable.len() {
                gram[a * persistable.len() + b] = (0..y.len())
                    .map(|i| oof_cols[a][i] as f64 * oof_cols[b][i] as f64)
                    .sum();
            }
        }
        let w = crate::linalg::nnls_cd(&gram, &aty, persistable.len(), 300);
        if w.iter().sum::<f64>() > 1e-9 {
            let blend_oof = blend(&oof_cols, &w);
            let (blend_mean, blend_lcb) =
                fold_score_lcb(&blend_oof, y, folds, task, n_classes, LCB_T);
            let infer_cols: Vec<Vec<f32>> = persistable
                .iter()
                .map(|&i| members[i].infer.clone())
                .collect();
            let blend_infer = blend(&infer_cols, &w);
            emit(
                "starforge_regressor_blend",
                blend_mean,
                Some(task_score(task, yh, &blend_infer, n_classes)),
            );
            let mut member_plans = Vec::new();
            for (&idx, &weight) in persistable.iter().zip(w.iter()) {
                if weight.abs() <= 1e-12 {
                    continue;
                }
                if let Some(plan) = &plans[idx] {
                    member_plans.push(MemberPlan {
                        name: plan.name.clone(),
                        backend: plan.backend.clone(),
                        weight,
                        spec: plan.spec,
                        starforge: plan.starforge.clone(),
                        math: plan.math.clone(),
                    });
                }
            }
            if !member_plans.is_empty() {
                blend_plan = Some(FuserPlan {
                    name: "starforge_regressor_blend".into(),
                    mode: super::super::FuserMode::WeightedBlend,
                    rank_outputs: false,
                    members: member_plans,
                    residual: None,
                });
                members.push(MemberPred {
                    name: "starforge_regressor_blend".into(),
                    oof: blend_oof,
                    infer: blend_infer,
                    fold_auc: vec![],
                    mean_auc: blend_mean,
                    lcb_auc: blend_lcb,
                });
                plans.push(None);
            }
        }
    }

    let mut best_i = 0usize;
    for i in 1..members.len() {
        if members[i].lcb_auc > members[best_i].lcb_auc + 1e-12 {
            best_i = i;
        }
    }
    let m = members.swap_remove(best_i);
    let selected_plan = if m.name == "starforge_regressor_blend" {
        blend_plan
    } else {
        plans.get(best_i).cloned().flatten().map(|mut plan| {
            plan.weight = 1.0;
            FuserPlan {
                name: m.name.clone(),
                mode: super::super::FuserMode::WeightedBlend,
                rank_outputs: false,
                members: vec![plan],
                residual: None,
            }
        })
    };

    let final_holdout = task_score(task, yh, &m.infer, n_classes);
    EnsembleResult {
        final_model: m.name.clone(),
        final_cv_auc: m.mean_auc,
        final_holdout_auc: final_holdout,
        final_holdout_preds: m.infer,
        catboost_cv,
        xgb_cv: catboost_cv,
        lgb_cv: catboost_cv,
        ensemble_cv: m.mean_auc,
        oof_lcb: m.lcb_auc,
        active_members: selected_plan.as_ref().map(|p| p.members.len()).unwrap_or(1),
        selected_fuser: m.name.clone(),
        family: "single-target".into(),
        single_lcb: m.lcb_auc,
        multi_lcb: f64::NAN,
        gate_passed: false,
        sibling_count: 0,
        final_plan: selected_plan,
    }
}
