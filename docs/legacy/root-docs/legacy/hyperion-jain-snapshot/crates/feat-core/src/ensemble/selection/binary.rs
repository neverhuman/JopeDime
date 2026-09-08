use super::super::ges::{bagged_ges, onehot, GesConfig};
use super::super::oof::build_oof_member;
use super::super::starforge::{append_starforge_members, residual_starforge_predictions};
use super::super::{
    blend, rank01_per_fold, EnsembleResult, FuserMode, FuserPlan, MemberPlan, MemberPred,
    ResidualFuserPlan, LCB_T,
};
use crate::backend::{Backend, TrainSpec};
use crate::data::Matrix;
use crate::metrics::{fold_auc_lcb, logloss, rank01, task_score};
use crate::progress::{Event, Sink};
use crate::starforge_integration::StarforgeConfig;
use crate::task::TaskKind;
use std::time::Instant;

const RESIDUAL_CLAMP_MIN: f32 = 1e-6;
const RESIDUAL_CLAMP_MAX: f32 = 1.0 - 1e-6;

struct FinalCandidate {
    name: String,
    lcb: f64,
    mean: f64,
    active: usize,
    infer: Vec<f32>,
    plan: FuserPlan,
}

#[allow(clippy::too_many_arguments)]
pub(super) fn select_binary_or_multilabel(
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

    let mut cat_s2 = *cv_spec;
    cat_s2.seed = cv_spec.seed ^ 0x9E37_79B9;
    let mut cat_h2 = *ho_spec;
    cat_h2.seed = ho_spec.seed ^ 0x9E37_79B9;

    // Build each OOF member and STREAM it live (with its per-fold CV scores) the moment it finishes, so
    // the cockpit leaderboard + CV heatmap fill in model-by-model rather than all appearing at once.
    let emit_member = |mp: &MemberPred| {
        let mut e = Event::new(phase, "model", t0.elapsed().as_secs_f64());
        e.model = Some(mp.name.clone());
        e.metric = Some(metric.to_string());
        e.best_val_auc = Some(mp.mean_auc);
        e.best_holdout_auc = Some(task_score(task, yh, &mp.infer, n_classes));
        e.folds = Some(mp.fold_auc.clone());
        sink.emit(&e);
    };
    let member_specs: [(&str, &dyn Backend, &TrainSpec, &TrainSpec); 4] = [
        ("catboost", cat, cv_spec, ho_spec),
        ("xgboost", xgb, cv_spec, ho_spec),
        ("lightgbm", lgb, cv_spec, ho_spec),
        ("catboost_s2", cat, &cat_s2, &cat_h2),
    ];
    let n_core = member_specs.len();
    let mut members: Vec<MemberPred> = Vec::with_capacity(n_core);
    for (name, be_ref, cvs, hos) in member_specs {
        let m = build_oof_member(
            name, be_ref, x, y, xh, folds, cols, cvs, hos, task, n_classes,
        );
        emit_member(&m);
        members.push(m);
    }
    let mut member_plan_specs: Vec<MemberPlan> = vec![
        MemberPlan {
            name: "catboost".into(),
            backend: "catboost".into(),
            weight: 0.0,
            spec: *ho_spec,
            starforge: None,
            math: None,
        },
        MemberPlan {
            name: "xgboost".into(),
            backend: "xgboost".into(),
            weight: 0.0,
            spec: *ho_spec,
            starforge: None,
            math: None,
        },
        MemberPlan {
            name: "lightgbm".into(),
            backend: "lightgbm".into(),
            weight: 0.0,
            spec: *ho_spec,
            starforge: None,
            math: None,
        },
        MemberPlan {
            name: "catboost_s2".into(),
            backend: "catboost".into(),
            weight: 0.0,
            spec: cat_h2,
            starforge: None,
            math: None,
        },
    ];
    append_starforge_members(
        &mut members,
        &mut member_plan_specs,
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
    // The 4 core members already streamed as they built; stream any appended Starforge members too.
    for mp in &members[n_core..] {
        emit_member(mp);
    }
    let (catboost_cv, xgb_cv, lgb_cv) = (
        members[0].mean_auc,
        members[1].mean_auc,
        members[2].mean_auc,
    );

    let cfg = GesConfig::default();
    let oof_cols: Vec<Vec<f32>> = members
        .iter()
        .map(|m| {
            if cfg.use_rank {
                rank01_per_fold(&m.oof, folds)
            } else {
                m.oof.clone()
            }
        })
        .collect();
    let infer_cols: Vec<Vec<f32>> = members
        .iter()
        .map(|m| {
            if cfg.use_rank {
                rank01(&m.infer)
            } else {
                m.infer.clone()
            }
        })
        .collect();

    let mut best_i = 0usize;
    for i in 1..members.len() {
        if members[i].lcb_auc > members[best_i].lcb_auc + 1e-12 {
            best_i = i;
        }
    }
    let mut sv_w = vec![0.0; members.len()];
    for w in sv_w.iter_mut().take(3) {
        *w = 1.0 / 3.0;
    }
    let sv = fold_auc_lcb(&blend(&oof_cols, &sv_w), y, folds, LCB_T);
    let sv_infer = blend(&infer_cols, &sv_w);
    emit(
        "ensemble",
        sv.mean,
        Some(task_score(task, yh, &sv_infer, n_classes)),
    );
    let ensemble_cv = sv.mean;
    let (ges_w, ges_mean, ges_lcb, ges_active) = bagged_ges(&oof_cols, y, folds, &cfg);
    let ges_infer = blend(&infer_cols, &ges_w);
    emit(
        "ges",
        ges_mean,
        Some(task_score(task, yh, &ges_infer, n_classes)),
    );

    let mut cand: Vec<FinalCandidate> = vec![
        weighted_candidate(
            members[best_i].name.clone(),
            members[best_i].lcb_auc,
            onehot(members.len(), best_i),
            members[best_i].mean_auc,
            1,
            &infer_cols,
            &member_plan_specs,
            cfg.use_rank,
        ),
        weighted_candidate(
            "ensemble".into(),
            sv.lcb,
            sv_w.clone(),
            sv.mean,
            3,
            &infer_cols,
            &member_plan_specs,
            cfg.use_rank,
        ),
        weighted_candidate(
            "ges".into(),
            ges_lcb,
            ges_w.clone(),
            ges_mean,
            ges_active,
            &infer_cols,
            &member_plan_specs,
            cfg.use_rank,
        ),
    ];
    if task == TaskKind::Binary {
        if let Some(residual) = residual_candidate(
            &members[0],
            &member_plan_specs[0],
            x,
            y,
            xh,
            folds,
            cols,
            ho_spec,
            starforge_cfg,
        ) {
            if residual.lcb > members[0].lcb_auc + 1e-9 {
                emit(
                    &residual.name,
                    residual.mean,
                    Some(task_score(task, yh, &residual.infer, n_classes)),
                );
                cand.push(residual);
            } else {
                eprintln!(
                    "[starforge] residual fusion skipped: lcb={:.5} did not beat catboost lcb={:.5}",
                    residual.lcb, members[0].lcb_auc
                );
            }
        }
    }
    let mut bi = 0usize;
    for i in 1..cand.len() {
        if cand[i].lcb > cand[bi].lcb + 1e-9 {
            bi = i;
        }
    }
    let selected = cand.swap_remove(bi);

    let final_holdout_auc = task_score(task, yh, &selected.infer, n_classes);
    EnsembleResult {
        final_model: selected.name.clone(),
        final_cv_auc: selected.mean,
        final_holdout_auc,
        final_holdout_preds: selected.infer,
        catboost_cv,
        xgb_cv,
        lgb_cv,
        ensemble_cv,
        oof_lcb: selected.lcb,
        active_members: selected.active,
        selected_fuser: selected.name,
        family: "single-target".into(),
        single_lcb: selected.lcb,
        multi_lcb: f64::NAN,
        gate_passed: false,
        sibling_count: 0,
        final_plan: Some(selected.plan),
    }
}

fn weighted_candidate(
    name: String,
    lcb: f64,
    weights: Vec<f64>,
    mean: f64,
    active: usize,
    infer_cols: &[Vec<f32>],
    member_plan_specs: &[MemberPlan],
    rank_outputs: bool,
) -> FinalCandidate {
    let infer = blend(infer_cols, &weights);
    let plan = FuserPlan {
        name: name.clone(),
        mode: FuserMode::WeightedBlend,
        rank_outputs,
        members: member_plan_specs
            .iter()
            .zip(weights.iter())
            .filter(|(_, &weight)| weight.abs() > 1e-12)
            .map(|(plan, &weight)| MemberPlan {
                name: plan.name.clone(),
                backend: plan.backend.clone(),
                weight,
                spec: plan.spec,
                starforge: plan.starforge.clone(),
                math: plan.math.clone(),
            })
            .collect(),
        residual: None,
    };
    FinalCandidate {
        name,
        lcb,
        mean,
        active,
        infer,
        plan,
    }
}

#[allow(clippy::too_many_arguments)]
fn residual_candidate(
    cat: &MemberPred,
    cat_plan: &MemberPlan,
    x: &Matrix,
    y: &[f32],
    xh: &Matrix,
    folds: &[(Vec<usize>, Vec<usize>)],
    cols: &[u32],
    ho_spec: &TrainSpec,
    cfg: &StarforgeConfig,
) -> Option<FinalCandidate> {
    let residual: Vec<f32> = y
        .iter()
        .zip(cat.oof.iter())
        .map(|(&yy, &pp)| yy - pp)
        .collect();
    let sf = residual_starforge_predictions(x, &residual, xh, folds, cols, cfg)?;
    let (alpha, mean, lcb, _) = tune_alpha(y, &cat.oof, &sf.oof, folds);
    let infer = fuse_residual(&cat.infer, &sf.infer, alpha);
    let residual_member = "starforge_residual_regressor".to_string();
    let base_member = cat.name.clone();
    let plan = FuserPlan {
        name: "catboost_residual_starforge".into(),
        mode: FuserMode::Residual,
        rank_outputs: false,
        members: vec![
            MemberPlan {
                name: base_member.clone(),
                backend: cat_plan.backend.clone(),
                weight: 1.0,
                spec: cat_plan.spec,
                starforge: None,
                math: cat_plan.math.clone(),
            },
            MemberPlan {
                name: residual_member.clone(),
                backend: "starforge".into(),
                weight: 1.0,
                spec: *ho_spec,
                starforge: Some(sf.plan),
                math: None,
            },
        ],
        residual: Some(ResidualFuserPlan {
            base_member,
            residual_member,
            alpha,
            clamp_min: RESIDUAL_CLAMP_MIN,
            clamp_max: RESIDUAL_CLAMP_MAX,
        }),
    };
    Some(FinalCandidate {
        name: "catboost_residual_starforge".into(),
        lcb,
        mean,
        active: 2,
        infer,
        plan,
    })
}

fn tune_alpha(
    y: &[f32],
    base_oof: &[f32],
    residual_oof: &[f32],
    folds: &[(Vec<usize>, Vec<usize>)],
) -> (f64, f64, f64, Vec<f32>) {
    let mut best_alpha: f64 = 0.0;
    let mut best_pred = fuse_residual(base_oof, residual_oof, 0.0);
    let mut best = fold_auc_lcb(&best_pred, y, folds, LCB_T);
    let mut best_logloss = logloss(y, &best_pred);
    for step in -20..=20 {
        let alpha = step as f64 * 0.05;
        let pred = fuse_residual(base_oof, residual_oof, alpha);
        let summ = fold_auc_lcb(&pred, y, folds, LCB_T);
        let ll = logloss(y, &pred);
        if summ.lcb > best.lcb + 1e-12
            || ((summ.lcb - best.lcb).abs() <= 1e-12
                && (ll < best_logloss - 1e-12
                    || ((ll - best_logloss).abs() <= 1e-12 && alpha.abs() < best_alpha.abs())))
        {
            best_alpha = alpha;
            best_pred = pred;
            best = summ;
            best_logloss = ll;
        }
    }
    (best_alpha, best.mean, best.lcb, best_pred)
}

fn fuse_residual(base: &[f32], residual: &[f32], alpha: f64) -> Vec<f32> {
    base.iter()
        .zip(residual)
        .map(|(&b, &r)| (b + (alpha as f32) * r).clamp(RESIDUAL_CLAMP_MIN, RESIDUAL_CLAMP_MAX))
        .collect()
}
