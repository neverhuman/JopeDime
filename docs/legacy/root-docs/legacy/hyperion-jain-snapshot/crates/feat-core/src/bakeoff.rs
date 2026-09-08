//! Phase-3.5 strategy bake-off.
//!
//! Instead of committing to one selection path, evaluate several candidate (model × feature-set)
//! strategies under the SAME stratified 5-fold CV, ship the CV winner as the champion, and record
//! every strategy's held-out *test* AUC so we can review whether CV-selection chose well.
//!
//! The five strategies (binary classification):
//!   1. `catboost+raw`   — CatBoost on the raw base features
//!   2. `catboost+gp`    — CatBoost on the reduced (base + top-10% GP) features
//!   3. `starforge+raw`  — Starforge (our in-house ICL classifier) on raw features
//!   4. `starforge+gp`   — Starforge on the reduced features
//!   5. `dominance+gp`   — the BBC-gated dominance ensemble on the reduced features
//!
//! Everything is reused: `SingleScorer` (CatBoost), `StarforgeSession::candidate` (Starforge), and
//! `select_dominant` (ensemble), all over the folds already drawn by the pipeline.
use std::time::Instant;

use crate::backend::Backend;
use crate::data::{Dataset, Matrix};
use crate::dominance::{select_dominant, DominanceConfig};
use crate::ensemble::EnsembleResult;
#[cfg(feature = "starforge-cpu")]
use crate::metrics::task_score;
use crate::phase3::SingleScorer;
use crate::pipeline::PipelineConfig;
use crate::progress::{Event, Sink};
use crate::select::MaskScorer;

/// Opt-in switch for the bake-off (wired from `--bakeoff`).
#[derive(Clone, Copy, Debug, Default)]
pub struct BakeoffConfig {
    pub enabled: bool,
}

/// One strategy's outcome. `cv_auc` drives selection; `test_auc` is logged for review.
#[derive(Clone, Debug)]
pub struct StrategyRow {
    pub name: String,
    pub features: usize,
    pub cv_auc: f64,
    pub cv_std: f64,
    pub test_auc: f64,
    /// false when a strategy was skipped (e.g. Starforge not compiled / weights missing).
    pub available: bool,
}

#[derive(Clone, Debug)]
pub struct BakeoffReport {
    pub rows: Vec<StrategyRow>,
    pub winner: usize,
}

/// The shipped winner plus the full report (for logging) and the ensemble plan if strategy 5 won.
pub struct BakeoffWinner {
    pub cols: Vec<u32>,
    pub cv_auc: f64,
    pub test_auc: f64,
    pub ensemble: Option<EnsembleResult>,
    pub report: BakeoffReport,
}

/// Starforge attention memory grows with feature width, so a wide GP set can OOM the GPU (≈63-feat
/// raw fits a shared 24GB card; ≈330-feat GP can OOM). Cap the columns fed to Starforge to the
/// top-K by importance percentile so strategies 3-4 stay runnable.
#[cfg(feature = "starforge-cpu")]
const STARFORGE_FEATURE_CAP: usize = 128;

#[cfg(feature = "starforge-cpu")]
fn cap_cols_by_importance(cols: &[u32], importance: &[f32], cap: usize) -> Vec<u32> {
    if cols.len() <= cap {
        return cols.to_vec();
    }
    let mut idx = cols.to_vec();
    idx.sort_by(|&a, &b| {
        let ia = importance.get(a as usize).copied().unwrap_or(0.0);
        let ib = importance.get(b as usize).copied().unwrap_or(0.0);
        ib.partial_cmp(&ia).unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.truncate(cap);
    idx
}

fn unavailable(name: &str) -> StrategyRow {
    StrategyRow {
        name: name.into(),
        features: 0,
        cv_auc: f64::NAN,
        cv_std: 0.0,
        test_auc: f64::NAN,
        available: false,
    }
}

fn warn(sink: &dyn Sink, t0: Instant, msg: &str) {
    let mut e = Event::new(5, "warning", t0.elapsed().as_secs_f64());
    e.model = Some(msg.to_string());
    sink.emit(&e);
}

fn emit_strategy(sink: &dyn Sink, t0: Instant, row: &StrategyRow, metric: &str) {
    let mut e = Event::new(5, "model", t0.elapsed().as_secs_f64());
    e.model = Some(row.name.clone());
    e.metric = Some(metric.to_string());
    if row.available {
        e.best_val_auc = Some(row.cv_auc);
        e.best_holdout_auc = Some(row.test_auc);
        e.best_size = Some(row.features as u64);
    }
    sink.emit(&e);
}

/// A single (backend, feature-set) strategy via the shared CV scorer: CV mean AUC + holdout AUC.
#[allow(clippy::too_many_arguments)]
fn single_strategy(
    name: &str,
    be: &dyn Backend,
    xtr_aug: &Matrix,
    ytr: &[f32],
    xho_aug: &Matrix,
    yh: &[f32],
    folds: &[(Vec<usize>, Vec<usize>)],
    cols: &[u32],
    cfg: &PipelineConfig,
) -> StrategyRow {
    let scorer = SingleScorer {
        x: xtr_aug,
        y: ytr,
        xh: xho_aug,
        yh,
        folds,
        be,
        cv_spec: cfg.cv_spec,
        ho_spec: cfg.ho_spec,
        task: cfg.task,
        n_classes: cfg.n_classes,
    };
    let (cv, std) = scorer.score_cv(cols);
    let test = scorer.score_holdout(cols);
    StrategyRow {
        name: name.into(),
        features: cols.len(),
        cv_auc: cv,
        cv_std: std,
        test_auc: test,
        available: true,
    }
}

/// Strategies 3 & 4: Starforge ICL on raw and reduced features. Compiled only with `starforge-cpu`.
#[cfg(feature = "starforge-cpu")]
#[allow(clippy::too_many_arguments)]
fn starforge_strategies(
    train: &Dataset,
    hold: &Dataset,
    xtr_aug: &Matrix,
    xho_aug: &Matrix,
    base_cols: &[u32],
    reduced_cols: &[u32],
    importance: &[f32],
    folds: &[(Vec<usize>, Vec<usize>)],
    cfg: &PipelineConfig,
    sink: &dyn Sink,
    t0: Instant,
) -> Vec<StrategyRow> {
    use crate::starforge_integration::load_session_for_task;
    let session = match load_session_for_task(&cfg.starforge, cfg.task) {
        Ok(Some(s)) => s,
        Ok(None) => {
            warn(
                sink,
                t0,
                "starforge unavailable (mode off or weights missing); skipping strategies 3-4",
            );
            return vec![unavailable("starforge+raw"), unavailable("starforge+gp")];
        }
        Err(e) => {
            warn(
                sink,
                t0,
                &format!("starforge load failed: {e}; skipping strategies 3-4"),
            );
            return vec![unavailable("starforge+raw"), unavailable("starforge+gp")];
        }
    };
    let run = |name: &str, cols: &[u32]| -> StrategyRow {
        match session.candidate(
            name,
            xtr_aug,
            &train.y,
            xho_aug,
            &hold.y,
            folds,
            cols,
            &cfg.starforge,
            cfg.task,
            cfg.n_classes,
        ) {
            Ok(c) => {
                let test = task_score(cfg.task, &hold.y, &c.infer, cfg.n_classes);
                let k = c.fold_auc.len().max(1) as f64;
                let var = c
                    .fold_auc
                    .iter()
                    .map(|a| (a - c.mean_auc) * (a - c.mean_auc))
                    .sum::<f64>()
                    / k;
                StrategyRow {
                    name: name.into(),
                    features: cols.len(),
                    cv_auc: c.mean_auc,
                    cv_std: var.sqrt(),
                    test_auc: test,
                    available: true,
                }
            }
            Err(e) => {
                warn(sink, t0, &format!("starforge {name} failed: {e}"));
                unavailable(name)
            }
        }
    };
    let raw_cols = cap_cols_by_importance(base_cols, importance, STARFORGE_FEATURE_CAP);
    let gp_cols = cap_cols_by_importance(reduced_cols, importance, STARFORGE_FEATURE_CAP);
    vec![
        run("starforge+raw", &raw_cols),
        run("starforge+gp", &gp_cols),
    ]
}

/// Stub when Starforge is not compiled in: both strategies are simply unavailable.
#[cfg(not(feature = "starforge-cpu"))]
#[allow(clippy::too_many_arguments)]
fn starforge_strategies(
    _train: &Dataset,
    _hold: &Dataset,
    _xtr_aug: &Matrix,
    _xho_aug: &Matrix,
    _base_cols: &[u32],
    _reduced_cols: &[u32],
    _importance: &[f32],
    _folds: &[(Vec<usize>, Vec<usize>)],
    _cfg: &PipelineConfig,
    sink: &dyn Sink,
    t0: Instant,
) -> Vec<StrategyRow> {
    warn(
        sink,
        t0,
        "starforge not compiled (rebuild with --features starforge-cuda or starforge-cpu); skipping strategies 3-4",
    );
    vec![unavailable("starforge+raw"), unavailable("starforge+gp")]
}

/// Strategy 5: the BBC-gated dominance ensemble on the reduced features.
#[allow(clippy::too_many_arguments)]
fn dominance_strategy(
    cat: &dyn Backend,
    extra: &[&dyn Backend],
    train: &Dataset,
    hold: &Dataset,
    xtr_aug: &Matrix,
    xho_aug: &Matrix,
    reduced_cols: &[u32],
    folds: &[(Vec<usize>, Vec<usize>)],
    cfg: &PipelineConfig,
    sink: &dyn Sink,
    t0: Instant,
) -> (StrategyRow, Option<EnsembleResult>) {
    let dcfg = DominanceConfig {
        enabled: true,
        ..cfg.dominance
    };
    let ens = select_dominant(
        cat,
        extra[0],
        extra[1],
        xtr_aug,
        &train.y,
        xho_aug,
        &hold.y,
        folds,
        reduced_cols,
        &cfg.cv_spec,
        &cfg.ho_spec,
        cfg.task,
        cfg.n_classes,
        &dcfg,
        sink,
        t0,
    );
    let row = StrategyRow {
        name: "dominance+gp".into(),
        features: reduced_cols.len(),
        cv_auc: ens.final_cv_auc,
        cv_std: 0.0,
        test_auc: ens.final_holdout_auc,
        available: true,
    };
    (row, Some(ens))
}

/// Run the 5-strategy bake-off. Caller guarantees `extra.len() >= 2` (xgb, lgb) and binary task.
#[allow(clippy::too_many_arguments)]
pub fn run_bakeoff(
    cat: &dyn Backend,
    extra: &[&dyn Backend],
    train: &Dataset,
    hold: &Dataset,
    xtr_aug: &Matrix,
    xho_aug: &Matrix,
    base_cols: &[u32],
    reduced_cols: &[u32],
    importance: &[f32],
    folds: &[(Vec<usize>, Vec<usize>)],
    cfg: &PipelineConfig,
    sink: &dyn Sink,
    t0: Instant,
) -> BakeoffWinner {
    let mut start = Event::new(5, "start", t0.elapsed().as_secs_f64());
    start.model = Some("bakeoff".into());
    sink.emit(&start);

    let mut rows: Vec<StrategyRow> = Vec::with_capacity(5);
    // 1, 2 — CatBoost on raw / reduced.
    rows.push(single_strategy(
        "catboost+raw",
        cat,
        xtr_aug,
        &train.y,
        xho_aug,
        &hold.y,
        folds,
        base_cols,
        cfg,
    ));
    rows.push(single_strategy(
        "catboost+gp",
        cat,
        xtr_aug,
        &train.y,
        xho_aug,
        &hold.y,
        folds,
        reduced_cols,
        cfg,
    ));
    // 3, 4 — Starforge on raw / reduced (feature-gated; unavailable rows if not compiled/loaded).
    rows.extend(starforge_strategies(
        train,
        hold,
        xtr_aug,
        xho_aug,
        base_cols,
        reduced_cols,
        importance,
        folds,
        cfg,
        sink,
        t0,
    ));
    // 5 — dominance ensemble on reduced.
    let (dom_row, dom_ens) = dominance_strategy(
        cat,
        extra,
        train,
        hold,
        xtr_aug,
        xho_aug,
        reduced_cols,
        folds,
        cfg,
        sink,
        t0,
    );
    rows.push(dom_row);

    let metric = crate::metrics::metric_name(cfg.task);
    for r in &rows {
        emit_strategy(sink, t0, r, metric);
    }

    // Winner = highest mean CV AUC among available strategies ("best k-fold").
    let winner = rows
        .iter()
        .enumerate()
        .filter(|(_, r)| r.available && r.cv_auc.is_finite())
        .max_by(|a, b| {
            a.1.cv_auc
                .partial_cmp(&b.1.cv_auc)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(i, _)| i)
        .unwrap_or(0);

    let win_name = rows[winner].name.clone();
    let win_cv = rows[winner].cv_auc;
    let win_test = rows[winner].test_auc;
    let win_feats = rows[winner].features;
    let cols = match win_name.as_str() {
        "catboost+raw" | "starforge+raw" => base_cols.to_vec(),
        _ => reduced_cols.to_vec(),
    };
    let ensemble = if win_name == "dominance+gp" {
        dom_ens
    } else {
        None
    };

    let mut done = Event::new(5, "done", t0.elapsed().as_secs_f64());
    done.model = Some(format!("bakeoff-winner={win_name}"));
    done.best_val_auc = Some(win_cv);
    done.best_holdout_auc = Some(win_test);
    done.best_size = Some(win_feats as u64);
    sink.emit(&done);

    BakeoffWinner {
        cols,
        cv_auc: win_cv,
        test_auc: win_test,
        ensemble,
        report: BakeoffReport { rows, winner },
    }
}
