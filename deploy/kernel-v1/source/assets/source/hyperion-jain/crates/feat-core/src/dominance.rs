//! The ensemble-dominance engine (`select_dominant`). On the champion feature space it builds a DIVERSE
//! OOF library — GBDTs (CatBoost/XGBoost/LightGBM + a seed variant) plus pure-Rust GLM (linear bias) and
//! RFF (smooth-kernel bias) — then chooses among {best-single, soft-vote, GES, logistic-stacker, BMA} by
//! **BBC-CV** (winner's-curse-corrected). Because best-single is one of the candidates, a blend is selected
//! ONLY if it honestly beats the best single member → "beat any single (constituent) model" by construction.
//! Classification only for now; regression/multiclass fall back to `select_model`. Default OFF.
use crate::backend::{Backend, TrainSpec};
use crate::data::Matrix;
use crate::ensemble::{
    bagged_ges, blend, build_oof_member, rank01_per_fold, select_model, EnsembleResult, GesConfig,
    MemberPred,
};
use crate::learners::glm::GlmBackend;
use crate::learners::rff::RffBackend;
use crate::meta::{bbc_cv_select, bma_weights};
use crate::metrics::{rank01, task_score};
use crate::progress::{Event, Sink};
use crate::task::TaskKind;
use std::time::Instant;

#[derive(Clone, Copy, Debug)]
pub struct DominanceConfig {
    pub enabled: bool,
    pub bbc_bootstraps: usize,
    pub seed: u64,
}
impl Default for DominanceConfig {
    fn default() -> Self {
        DominanceConfig {
            enabled: false,
            bbc_bootstraps: 200,
            seed: 20260616,
        }
    }
}

fn flatten_cols(cols: &[Vec<f32>], n: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(cols.len() * n);
    for c in cols {
        out.extend_from_slice(&c[..n.min(c.len())]);
    }
    out
}

#[allow(clippy::too_many_arguments)]
pub fn select_dominant(
    cat: &dyn Backend,
    xgb: &dyn Backend,
    lgb: &dyn Backend,
    x: &Matrix,
    y: &[f32],
    xh: &Matrix,
    yh: &[f32],
    folds: &[(Vec<usize>, Vec<usize>)],
    cols: &[u32],
    cv: &TrainSpec,
    ho: &TrainSpec,
    task: TaskKind,
    n_classes: usize,
    dcfg: &DominanceConfig,
    sink: &dyn Sink,
    t0: Instant,
) -> EnsembleResult {
    // Classification only; regression/multiclass dominance comes with the BMA backend (C4).
    if !matches!(task, TaskKind::Binary | TaskKind::MultiLabel) {
        let disabled = crate::starforge_integration::StarforgeConfig::disabled();
        return select_model(
            cat,
            xgb,
            lgb,
            x,
            y,
            xh,
            yh,
            folds,
            cols,
            cv,
            ho,
            task,
            n_classes,
            &disabled,
            &[],
            None,
            sink,
            t0,
        );
    }
    let phase = 5u8;
    let mut e = Event::new(phase, "start", t0.elapsed().as_secs_f64());
    e.final_features = Some(cols.len() as u64);
    sink.emit(&e);
    let emit = |name: &str, v: f64| {
        let mut e = Event::new(phase, "model", t0.elapsed().as_secs_f64());
        e.model = Some(name.into());
        e.best_val_auc = Some(v);
        sink.emit(&e);
    };

    let mut cat_s2 = *cv;
    cat_s2.seed ^= 0x9E37_79B9;
    let mut cat_h2 = *ho;
    cat_h2.seed ^= 0x9E37_79B9;
    let glm = GlmBackend::logistic();
    let rff = RffBackend::logistic();

    // The diverse OOF library — each is a fold-pure member via build_oof_member (inherits the leak proofs).
    let members: Vec<MemberPred> = vec![
        build_oof_member(
            "catboost", cat, x, y, xh, folds, cols, cv, ho, task, n_classes,
        ),
        build_oof_member(
            "xgboost", xgb, x, y, xh, folds, cols, cv, ho, task, n_classes,
        ),
        build_oof_member(
            "lightgbm", lgb, x, y, xh, folds, cols, cv, ho, task, n_classes,
        ),
        build_oof_member(
            "catboost_s2",
            cat,
            x,
            y,
            xh,
            folds,
            cols,
            &cat_s2,
            &cat_h2,
            task,
            n_classes,
        ),
        build_oof_member("glm", &glm, x, y, xh, folds, cols, cv, ho, task, n_classes),
        build_oof_member("rff", &rff, x, y, xh, folds, cols, cv, ho, task, n_classes),
    ];
    for m in &members {
        emit(&m.name, m.mean_auc);
    }
    let mn = members.len();
    let best_i = {
        let mut bi = 0usize;
        for i in 1..mn {
            if members[i].lcb_auc > members[bi].lcb_auc + 1e-12 {
                bi = i;
            }
        }
        bi
    };
    let (catboost_cv, xgb_cv, lgb_cv) = (
        members[0].mean_auc,
        members[1].mean_auc,
        members[2].mean_auc,
    );

    // Rank-normalized columns (per-fold for OOF, global for holdout) so heterogeneous member ranges combine.
    let oof_cols: Vec<Vec<f32>> = members
        .iter()
        .map(|m| rank01_per_fold(&m.oof, folds))
        .collect();
    let infer_cols: Vec<Vec<f32>> = members.iter().map(|m| rank01(&m.infer)).collect();

    // Logistic stacker over the OOF matrix (itself a fold-pure member ⇒ honest OOF).
    let (n, nh) = (y.len(), yh.len());
    let z = Matrix::new(flatten_cols(&oof_cols, n), n, mn);
    let zh = Matrix::new(flatten_cols(&infer_cols, nh), nh, mn);
    let zcols: Vec<u32> = (0..mn as u32).collect();
    let stacker = build_oof_member(
        "stacker", &glm, &z, y, &zh, folds, &zcols, cv, ho, task, n_classes,
    );
    emit("stacker", stacker.mean_auc);

    // Candidates (best-single FIRST so ties favour the simplest model).
    let sv_w = vec![1.0 / mn as f64; mn];
    let (ges_w, _, _, _) = bagged_ges(&oof_cols, y, folds, &GesConfig::default());
    let bma_w = bma_weights(&oof_cols, y, 50);
    let cands: Vec<(String, Vec<f32>, Vec<f32>)> = vec![
        (
            "best_single".into(),
            members[best_i].oof.clone(),
            members[best_i].infer.clone(),
        ),
        (
            "soft_vote".into(),
            blend(&oof_cols, &sv_w),
            blend(&infer_cols, &sv_w),
        ),
        (
            "ges".into(),
            blend(&oof_cols, &ges_w),
            blend(&infer_cols, &ges_w),
        ),
        ("stacker".into(), stacker.oof.clone(), stacker.infer.clone()),
        (
            "bma".into(),
            blend(&oof_cols, &bma_w),
            blend(&infer_cols, &bma_w),
        ),
    ];
    let cand_oof: Vec<Vec<f32>> = cands.iter().map(|c| c.1.clone()).collect();
    let bbc = bbc_cv_select(
        &cand_oof,
        y,
        task,
        n_classes,
        dcfg.bbc_bootstraps,
        dcfg.seed,
    );
    let win = bbc.winner;
    let beat_best = win != 0; // a blend was honestly chosen over the best single member
    emit(&format!("BBC:{}", cands[win].0), bbc.corrected);

    let final_holdout = task_score(task, yh, &cands[win].2, n_classes);
    {
        let mut e = Event::new(phase, "done", t0.elapsed().as_secs_f64());
        e.model = Some(cands[win].0.clone());
        e.best_val_auc = Some(bbc.corrected);
        e.best_holdout_auc = Some(final_holdout);
        sink.emit(&e);
    }
    EnsembleResult {
        final_model: cands[win].0.clone(),
        final_cv_auc: bbc.corrected,
        final_holdout_auc: final_holdout,
        final_holdout_preds: cands[win].2.clone(),
        catboost_cv,
        xgb_cv,
        lgb_cv,
        ensemble_cv: bbc.corrected,
        oof_lcb: bbc.ci_lo,
        active_members: mn,
        selected_fuser: cands[win].0.clone(),
        family: "dominance".into(),
        single_lcb: members[best_i].lcb_auc,
        multi_lcb: bbc.corrected,
        gate_passed: beat_best,
        sibling_count: mn,
        final_plan: None,
    }
}
