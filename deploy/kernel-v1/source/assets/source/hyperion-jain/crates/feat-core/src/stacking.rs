//! Multi-target STACKING on the GP-reduced champion feature space `F`. Correlated outcomes "borrow
//! strength": sibling targets' fold-pure out-of-fold predictions become meta-features for the primary.
//!
//! Design thesis: every piece reduces to `ensemble::build_oof_member`, so it inherits that function's
//! leak proofs (Invariant P: per-row OOF purity; H: holdout firewall; C: coverage). The one new
//! invariant is **S (shared folds)** — sibling OOF and the primary meta-CV use the *same* `folds`, so a
//! sibling meta-feature at a primary validation row is always out-of-fold for it.
//!
//! Two cross-target candidates (both via build_oof_member):
//!   - GBDT stacker: a shallow CatBoost over [primary-OOF | sibling-OOF].
//!   - Sibling-stacked primary: CatBoost over [F | sibling-OOF] (captures feature×sibling interactions).
//!
//! The multi-target path SHIPS ONLY IF its OOF-LCB beats the single-target ceiling + a margin (no
//! regression, fairness-firewalled). `k == 1` makes the whole layer inert.
use crate::backend::{Backend, TrainSpec};
use crate::data::Matrix;
use crate::ensemble::{
    build_oof_member, rank01_per_fold, select_model, EnsembleResult, MemberPred,
};
use crate::metrics::{rank01, task_score};
use crate::progress::{Event, Sink};
use crate::task::TaskKind;
use std::time::Instant;

#[derive(Clone)]
pub struct StackConfig {
    /// Master switch; auto-off when there are no siblings (k == 1).
    pub enabled: bool,
    /// Min OOF-LCB gain over the single-target ceiling required to ship the multi-target path.
    pub gate_margin: f64,
    /// Rank-normalize OOF/infer meta-columns (fold-local for OOF, global for infer) before stacking.
    pub rank_normalize: bool,
    pub seed: u64,
}
impl Default for StackConfig {
    fn default() -> Self {
        StackConfig {
            enabled: false,
            gate_margin: 1e-4,
            rank_normalize: true,
            seed: 20260616,
        }
    }
}

/// Per-sibling fold-pure OOF (length n) + full-train-refit holdout inference (length nh) on `F`.
pub struct SiblingLibrary {
    pub names: Vec<String>,
    pub oof: Vec<Vec<f32>>,
    pub infer: Vec<Vec<f32>>,
}

#[allow(clippy::too_many_arguments)]
pub fn build_sibling_oof_singles(
    be: &dyn Backend,
    x: &Matrix,
    y_multi: &[f32],
    k: usize,
    primary: usize,
    n: usize,
    xh: &Matrix,
    folds: &[(Vec<usize>, Vec<usize>)],
    cols: &[u32],
    cv_spec: &TrainSpec,
    ho_spec: &TrainSpec,
    names: &[String],
) -> SiblingLibrary {
    let (mut oof, mut infer, mut nm) = (Vec::new(), Vec::new(), Vec::new());
    for t in 0..k {
        if t == primary {
            continue;
        }
        let yt = &y_multi[t * n..(t + 1) * n]; // sibling labels are independent binaries
        let mp = build_oof_member(
            &format!("sib_{t}"),
            be,
            x,
            yt,
            xh,
            folds,
            cols,
            cv_spec,
            ho_spec,
            TaskKind::Binary,
            2,
        );
        oof.push(mp.oof);
        infer.push(mp.infer);
        nm.push(names.get(t).cloned().unwrap_or_else(|| format!("t{t}")));
    }
    SiblingLibrary {
        names: nm,
        oof,
        infer,
    }
}

/// rank-normalized column: fold-local for OOF (Invariant R), global for holdout inference.
fn norm_oof(col: &[f32], folds: &[(Vec<usize>, Vec<usize>)], on: bool) -> Vec<f32> {
    if on {
        rank01_per_fold(col, folds)
    } else {
        col.to_vec()
    }
}
fn norm_infer(col: &[f32], on: bool) -> Vec<f32> {
    if on {
        rank01(col)
    } else {
        col.to_vec()
    }
}

/// Build the level-1 meta design: columns = [primary model OOF, sibling-1 OOF, …]. Train + holdout.
fn meta_matrix(
    p_oof: &[f32],
    p_infer: &[f32],
    sib: &SiblingLibrary,
    folds: &[(Vec<usize>, Vec<usize>)],
    rn: bool,
) -> (Matrix, Matrix, usize) {
    let n = p_oof.len();
    let nh = p_infer.len();
    let m = 1 + sib.oof.len();
    let mut tr = Vec::with_capacity(n * m);
    let mut ho = Vec::with_capacity(nh * m);
    tr.extend_from_slice(&norm_oof(p_oof, folds, rn));
    ho.extend_from_slice(&norm_infer(p_infer, rn));
    for s in 0..sib.oof.len() {
        tr.extend_from_slice(&norm_oof(&sib.oof[s], folds, rn));
        ho.extend_from_slice(&norm_infer(&sib.infer[s], rn));
    }
    (Matrix::new(tr, n, m), Matrix::new(ho, nh, m), m)
}

/// Sibling-stacked primary member: CatBoost over [champion features F | sibling-OOF], leak-free via
/// shared folds. Captures feature×sibling interactions the flat meta-learner can't.
#[allow(clippy::too_many_arguments)]
pub fn build_sibling_stacked_member(
    name: &str,
    be: &dyn Backend,
    x: &Matrix,
    y: &[f32],
    xh: &Matrix,
    folds: &[(Vec<usize>, Vec<usize>)],
    cols: &[u32],
    sib: &SiblingLibrary,
    rank_normalize: bool,
    cv_spec: &TrainSpec,
    ho_spec: &TrainSpec,
) -> MemberPred {
    let (n, nh, p_champ, n_sib) = (x.n, xh.n, cols.len(), sib.oof.len());
    let mut tr_flat = x.select_cols(cols); // n × p_champ, column-major
    let mut ho_flat = xh.select_cols(cols);
    for s in 0..n_sib {
        tr_flat.extend_from_slice(&norm_oof(&sib.oof[s], folds, rank_normalize));
        ho_flat.extend_from_slice(&norm_infer(&sib.infer[s], rank_normalize));
    }
    let x_stack = Matrix::new(tr_flat, n, p_champ + n_sib);
    let xh_stack = Matrix::new(ho_flat, nh, p_champ + n_sib);
    let cols_stack: Vec<u32> = (0..(p_champ + n_sib) as u32).collect();
    build_oof_member(
        name,
        be,
        &x_stack,
        y,
        &xh_stack,
        folds,
        &cols_stack,
        cv_spec,
        ho_spec,
        TaskKind::Binary,
        2,
    )
}

/// Stacked model selection: build the single-target ceiling, then cross-target candidates, and ship the
/// multi-target path ONLY if it beats the ceiling by `gate_margin` (OOF-LCB). `k == 1` ⇒ inert.
#[allow(clippy::too_many_arguments)]
pub fn select_model_stacked(
    cat: &dyn Backend,
    xgb: &dyn Backend,
    lgb: &dyn Backend,
    x: &Matrix,
    y: &[f32],
    xh: &Matrix,
    yh: &[f32],
    y_multi: &[f32],
    k: usize,
    primary: usize,
    names: &[String],
    folds: &[(Vec<usize>, Vec<usize>)],
    cols: &[u32],
    cv_spec: &TrainSpec,
    ho_spec: &TrainSpec,
    task: TaskKind,
    n_classes: usize,
    scfg: &StackConfig,
    sink: &dyn Sink,
    t0: Instant,
) -> EnsembleResult {
    // Single-target ceiling (the bar the gate must beat).
    let disabled = crate::starforge_integration::StarforgeConfig::disabled();
    let mut single = select_model(
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
        &disabled,
        &[],
        None,
        sink,
        t0,
    );
    let single_lcb = single.oof_lcb;
    let sibling_count = k.saturating_sub(1);
    if k <= 1 {
        single.single_lcb = single_lcb;
        return single; // inert
    }

    let phase = 5u8;
    let emit = |name: &str, cv: f64| {
        let mut e = Event::new(phase, "model", t0.elapsed().as_secs_f64());
        e.model = Some(name.into());
        e.best_val_auc = Some(cv);
        sink.emit(&e);
    };

    // Primary model OOF (meta-feature) + sibling OOF library, all on the SAME folds (Invariant S).
    let pm = build_oof_member(
        "catboost", cat, x, y, xh, folds, cols, cv_spec, ho_spec, task, n_classes,
    );
    let sib = build_sibling_oof_singles(
        cat, x, y_multi, k, primary, x.n, xh, folds, cols, cv_spec, ho_spec, names,
    );

    // Candidate A — shallow GBDT stacker over [primary-OOF | sibling-OOF].
    let mut shallow = *cv_spec;
    shallow.depth = 3;
    let (mtr, mho, mcols) = meta_matrix(&pm.oof, &pm.infer, &sib, folds, scfg.rank_normalize);
    let meta_cols: Vec<u32> = (0..mcols as u32).collect();
    let gbdt = build_oof_member(
        "gbdt_stacker",
        cat,
        &mtr,
        y,
        &mho,
        folds,
        &meta_cols,
        &shallow,
        &shallow,
        TaskKind::Binary,
        2,
    );
    emit("gbdt_stacker", gbdt.mean_auc);

    // Candidate B — sibling-stacked primary over [F | sibling-OOF].
    let stacked = build_sibling_stacked_member(
        "sib_stacked",
        cat,
        x,
        y,
        xh,
        folds,
        cols,
        &sib,
        scfg.rank_normalize,
        cv_spec,
        ho_spec,
    );
    emit("sib_stacked", stacked.mean_auc);

    // Best multi candidate by OOF-LCB (ties favour the simpler GBDT stacker).
    let best = if stacked.lcb_auc > gbdt.lcb_auc + 1e-12 {
        stacked
    } else {
        gbdt
    };
    let multi_lcb = best.lcb_auc;
    let gate_passed = multi_lcb > single_lcb + scfg.gate_margin;

    if gate_passed {
        let final_holdout = task_score(task, yh, &best.infer, n_classes);
        let mut e = Event::new(phase, "done", t0.elapsed().as_secs_f64());
        e.model = Some(best.name.clone());
        e.best_val_auc = Some(best.mean_auc);
        e.best_holdout_auc = Some(final_holdout);
        sink.emit(&e);
        EnsembleResult {
            final_model: best.name.clone(),
            final_cv_auc: best.mean_auc,
            final_holdout_auc: final_holdout,
            final_holdout_preds: best.infer,
            catboost_cv: single.catboost_cv,
            xgb_cv: single.xgb_cv,
            lgb_cv: single.lgb_cv,
            ensemble_cv: single.ensemble_cv,
            oof_lcb: multi_lcb,
            active_members: 1,
            selected_fuser: best.name,
            family: "multi-target".into(),
            single_lcb,
            multi_lcb,
            gate_passed: true,
            sibling_count,
            final_plan: None,
        }
    } else {
        // Multi-target did not beat the single ceiling → keep the single-target selection unchanged.
        single.single_lcb = single_lcb;
        single.multi_lcb = multi_lcb;
        single.sibling_count = sibling_count;
        single
    }
}
