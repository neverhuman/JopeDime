//! Fit / predict / cross-validate a battle genome — the reference execution
//! path used for claim rescoring, calibration probes, and golden tests.

use crate::dsl::Stage;
use crate::folds::fold_of;
use crate::genome::GenomeIr;
use crate::heads::{
    fit_head, fit_head_and_predict_train_with_cache, fit_head_with_cache, is_head_op, FittedHead,
    RidgeModel,
};
use crate::kernel::width::{reject_zero_width_rows, WidthMismatch};
use crate::math_expr::MathExpr;
use crate::metrics::r2;
use crate::ops::{
    apply_op, apply_op_train, fit_op, fit_op_with_cache, is_append_noop_at_max_width, op_append,
    FittedOp, SortedColumnsCache,
};
use crate::target_transform::{fit_target, is_target_op, FittedTarget};
use crate::{sanitize, Rows};
use anyhow::{bail, Result};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub enum Model {
    Pipeline {
        /// Optional target-side transform (fitted on train y, inverted at
        /// predict); always the FIRST stage of the source pipeline.
        target: Option<FittedTarget>,
        ops: Vec<FittedOp>,
        head: Box<FittedHead>,
    },
    Symbolic {
        expr: MathExpr,
        a: f64,
        b: f64,
    },
    /// The explicit nested portfolio form (spec §2): member pipelines fitted
    /// and OOF-measured inside portfolio.rs. The compact head form stays a
    /// Pipeline whose terminal head is `FittedHead::Portfolio`.
    Portfolio(crate::portfolio::PortfolioModel),
}

pub fn fit(genome: &GenomeIr, x: &Rows, y: &[f64]) -> Result<Model> {
    if x.is_empty() || y.len() < x.len() {
        bail!("training data is empty or target is short");
    }
    reject_zero_width_rows(x)?;
    match genome {
        GenomeIr::Pipeline(dsl) => {
            let (target_stage, rest) = split_target(&dsl.stages)?;
            let (transform_stages, head_stage) = split_stages(rest);
            // The transform is fitted on the raw train target; everything
            // downstream (target-aware encoders included) sees only y'.
            let target = match target_stage {
                Some(stage) => Some(fit_target(stage, y)?),
                None => None,
            };
            let y_owned: Vec<f64>;
            let y_eff: &[f64] = match &target {
                Some(t) => {
                    y_owned = t.forward_all(y);
                    &y_owned
                }
                None => y,
            };
            let mut cur = x.clone();
            sanitize(&mut cur);
            let mut sorted_cache = SortedColumnsCache::new();
            let mut ops = Vec::with_capacity(transform_stages.len());
            for stage in transform_stages {
                let in_cols = cur.first().map_or(0, Vec::len);
                let op = fit_op_with_cache(stage, &cur, y_eff, &mut sorted_cache)?;
                if !is_append_noop_at_max_width(&op, &cur, true) {
                    cur = apply_op_train(&op, &cur)?;
                }
                let new_width = cur.first().map_or(0, Vec::len);
                sorted_cache.on_stage_complete(op_append(&op), in_cols, new_width);
                ops.push(op);
            }
            let head = Box::new(fit_head_with_cache(
                &head_stage,
                &cur,
                y_eff,
                &mut sorted_cache,
            )?);
            Ok(Model::Pipeline { target, ops, head })
        }
        GenomeIr::Portfolio(pir) => {
            // Explicit members carry neither gates nor view flags (both are
            // registry data): gate Always, raw view.
            let roster: Vec<crate::members::ResolvedMember> = pir
                .members
                .iter()
                .enumerate()
                .map(|(i, dsl)| {
                    crate::members::ResolvedMember::from_pipe(format!("member_{i}"), dsl.clone())
                })
                .collect();
            let model =
                crate::portfolio::fit_portfolio(&roster, x, y, pir.folds, pir.repeats_auto)?;
            Ok(Model::Portfolio(model))
        }
        GenomeIr::Symbolic { expr, affine, .. } => {
            let f: Vec<f64> = x
                .iter()
                .map(|row| expr.eval_row(row).unwrap_or(0.0))
                .map(|v| if v.is_finite() { v } else { 0.0 })
                .collect();
            let n = f.len() as f64;
            let (a, b) = if *affine {
                // Keijzer linear scaling: least-squares slope/intercept on train.
                let fm = f.iter().sum::<f64>() / n;
                let ym = y[..f.len()].iter().sum::<f64>() / n;
                let mut cov = 0.0;
                let mut var = 0.0;
                for i in 0..f.len() {
                    let d = f[i] - fm;
                    cov += d * (y[i] - ym);
                    var += d * d;
                }
                if var > 1e-24 {
                    let a = cov / var;
                    (a, ym - a * fm)
                } else {
                    (0.0, ym)
                }
            } else {
                (1.0, 0.0)
            };
            Ok(Model::Symbolic {
                expr: expr.clone(),
                a,
                b,
            })
        }
    }
}

/// Fits a `Model` and returns both the fitted `Model` and the training predictions
/// computed directly from the already-transformed training feature matrix `cur`
/// (avoiding a second redundant pass through `ops` in `model.predict(tx)`).
/// Bit-identical (`to_bits()`) to calling `fit(genome, x, y)` followed by `model.predict(x)`.
pub fn fit_and_predict_train(genome: &GenomeIr, x: &Rows, y: &[f64]) -> Result<(Model, Vec<f64>)> {
    if x.is_empty() || y.len() < x.len() {
        bail!("training data is empty or target is short");
    }
    reject_zero_width_rows(x)?;
    match genome {
        GenomeIr::Pipeline(dsl) => {
            let (target_stage, rest) = split_target(&dsl.stages)?;
            let (transform_stages, head_stage) = split_stages(rest);
            let target = match target_stage {
                Some(stage) => Some(fit_target(stage, y)?),
                None => None,
            };
            let y_owned: Vec<f64>;
            let y_eff: &[f64] = match &target {
                Some(t) => {
                    y_owned = t.forward_all(y);
                    &y_owned
                }
                None => y,
            };
            let mut cur = x.clone();
            sanitize(&mut cur);
            let mut sorted_cache = SortedColumnsCache::new();
            let mut ops = Vec::with_capacity(transform_stages.len());
            for stage in transform_stages {
                let in_cols = cur.first().map_or(0, Vec::len);
                let op = fit_op_with_cache(stage, &cur, y_eff, &mut sorted_cache)?;
                if !is_append_noop_at_max_width(&op, &cur, true) {
                    cur = apply_op_train(&op, &cur)?;
                }
                let new_width = cur.first().map_or(0, Vec::len);
                sorted_cache.on_stage_complete(op_append(&op), in_cols, new_width);
                ops.push(op);
            }
            let (head_val, raw_preds) =
                fit_head_and_predict_train_with_cache(&head_stage, &cur, y_eff, &mut sorted_cache)?;
            let head = Box::new(head_val);
            let tr_preds = match &target {
                None => raw_preds,
                Some(t) => raw_preds
                    .iter()
                    .map(|p| t.final_clamp(t.inverse(*p)))
                    .collect(),
            };
            Ok((Model::Pipeline { target, ops, head }, tr_preds))
        }
        _ => {
            let model = fit(genome, x, y)?;
            let tr_preds = model.predict(x)?;
            Ok((model, tr_preds))
        }
    }
}

/// `fit` with a stage-prefix cache: bit-identical to `fit` (the cache key
/// pins the entire computation, so a hit substitutes the output of the
/// identical work), but shared transform prefixes across genomes on the same
/// fold are fitted once. Target transforms and heads are always fitted
/// per-call; only transform stages and their TRAIN-pass blocks are cached.
/// Non-pipeline genomes delegate to plain `fit`.
pub fn fit_with_stage_cache(
    genome: &GenomeIr,
    x: &Rows,
    y: &[f64],
    cache: &crate::stage_cache::StageCache,
    ctx: &crate::stage_cache::StageCacheCtx,
) -> Result<Model> {
    if x.is_empty() || y.len() < x.len() {
        bail!("training data is empty or target is short");
    }
    reject_zero_width_rows(x)?;
    let GenomeIr::Pipeline(dsl) = genome else {
        return fit(genome, x, y);
    };
    let (target_stage, rest) = split_target(&dsl.stages)?;
    let (transform_stages, head_stage) = split_stages(rest);
    let target = match target_stage {
        Some(stage) => Some(fit_target(stage, y)?),
        None => None,
    };
    let y_owned: Vec<f64>;
    let y_eff: &[f64] = match &target {
        Some(t) => {
            y_owned = t.forward_all(y);
            &y_owned
        }
        None => y,
    };
    let mut key = cache.chain_root(ctx, target_stage);
    let mut cur: std::sync::Arc<Rows> = {
        let mut sanitized = x.clone();
        sanitize(&mut sanitized);
        std::sync::Arc::new(sanitized)
    };
    let mut ops = Vec::with_capacity(transform_stages.len());
    for stage in transform_stages {
        key = cache.chain_extend(&key, stage);
        if let Some((fitted, block)) = cache.get(&key) {
            ops.push(fitted);
            cur = block;
            continue;
        }
        let op = fit_op(stage, &cur, y_eff)?;
        let block = std::sync::Arc::new(apply_op_train(&op, &cur)?);
        cache.insert(key, op.clone(), std::sync::Arc::clone(&block));
        ops.push(op);
        cur = block;
    }
    let head = Box::new(fit_head(&head_stage, &cur, y_eff)?);
    Ok(Model::Pipeline { target, ops, head })
}

impl Model {
    pub fn predict(&self, x: &Rows) -> std::result::Result<Vec<f64>, WidthMismatch> {
        reject_zero_width_rows(x)?;
        Ok(match self {
            Model::Pipeline { target, ops, head } => {
                let mut cur = x.clone();
                sanitize(&mut cur);
                for (stage, op) in ops.iter().enumerate() {
                    if !is_append_noop_at_max_width(op, &cur, false) {
                        cur = apply_op(op, &cur).map_err(|mut error| {
                            error.stage = stage;
                            error
                        })?;
                    }
                }
                let preds = head.predict(&cur)?;
                match target {
                    None => preds,
                    // Exact inverse back to the original target scale, then
                    // the kind's final prediction clamp (±60 for legacy kinds,
                    // train-range for range-aware kinds).
                    Some(t) => preds.iter().map(|p| t.final_clamp(t.inverse(*p))).collect(),
                }
            }
            Model::Portfolio(m) => m.predict(x)?,
            Model::Symbolic { expr, a, b } => x
                .iter()
                .map(|row| {
                    let v = expr.eval_row(row).unwrap_or(0.0);
                    let v = if v.is_finite() { v } else { 0.0 };
                    let z = a * v + b;
                    if z.is_finite() {
                        z.clamp(-60.0, 60.0)
                    } else {
                        0.0
                    }
                })
                .collect(),
        })
    }

    /// Portfolio diagnostics (member OOF RMSEs, NNLS weights, blend stats)
    /// when this model is a portfolio — explicit form or compact head.
    pub fn portfolio_diagnostics(&self) -> Option<&crate::portfolio::PortfolioDiagnostics> {
        match self {
            Model::Portfolio(m) => Some(&m.diagnostics),
            Model::Pipeline { head, .. } => match head.as_ref() {
                crate::heads::FittedHead::Portfolio(m) => Some(&m.diagnostics),
                _ => None,
            },
            _ => None,
        }
    }
}

/// Split off the optional leading target pseudo-stage; a target stage
/// anywhere else in the pipeline is an error.
pub(crate) fn split_target(stages: &[Stage]) -> Result<(Option<&Stage>, &[Stage])> {
    let (target, rest) = match stages.first() {
        Some(first) if is_target_op(&first.op) => (Some(first), &stages[1..]),
        _ => (None, stages),
    };
    if rest.iter().any(|s| is_target_op(&s.op)) {
        bail!("target transform must be the first stage");
    }
    Ok((target, rest))
}

/// Split pipeline stages into transforms + terminal head; a missing head
/// defaults to plain ridge.
pub(crate) fn split_stages(stages: &[Stage]) -> (&[Stage], Stage) {
    if let Some(last) = stages.last() {
        if is_head_op(&last.op) {
            return (&stages[..stages.len() - 1], last.clone());
        }
    }
    (stages, Stage::new("ridge"))
}

pub fn fit_predict(
    genome: &GenomeIr,
    x_train: &Rows,
    y_train: &[f64],
    x_test: &Rows,
) -> Result<Vec<f64>> {
    let model = fit(genome, x_train, y_train)?;
    Ok(model.predict(x_test)?)
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct CvReport {
    pub fold_r2: Vec<f64>,
    pub mean_r2: f64,
    /// Pooled out-of-fold r2 — the primary CV estimate (more stable than the
    /// per-fold mean on 1000-row datasets).
    pub oof_r2: f64,
}

/// Honest k-fold CV: the WHOLE pipeline (fitted transforms, target-aware
/// encoders, head) is refit inside each fold. Fold assignment is the shared
/// deterministic `fold_of`, identical across every experiment.
pub fn cv_r2(
    genome: &GenomeIr,
    x: &Rows,
    y: &[f64],
    k: usize,
    dataset_key: &str,
) -> Result<CvReport> {
    let n = x.len().min(y.len());
    let k = k.max(2);
    let mut oof = vec![f64::NAN; n];
    let mut fold_r2 = Vec::with_capacity(k);
    for fold in 0..k {
        let mut xtr: Rows = Vec::new();
        let mut ytr: Vec<f64> = Vec::new();
        let mut xva: Rows = Vec::new();
        let mut va_idx: Vec<usize> = Vec::new();
        for r in 0..n {
            if fold_of(dataset_key, r, k) == fold {
                xva.push(x[r].clone());
                va_idx.push(r);
            } else {
                xtr.push(x[r].clone());
                ytr.push(y[r]);
            }
        }
        if xva.is_empty() || xtr.len() < 8 {
            continue;
        }
        let preds = fit_predict(genome, &xtr, &ytr, &xva)?;
        let yva: Vec<f64> = va_idx.iter().map(|&r| y[r]).collect();
        fold_r2.push(r2(&yva, &preds));
        for (i, &r) in va_idx.iter().enumerate() {
            oof[r] = preds[i];
        }
    }
    if fold_r2.is_empty() {
        bail!("cv produced no folds");
    }
    let mean_r2 = fold_r2.iter().sum::<f64>() / fold_r2.len() as f64;
    let mut ys = Vec::new();
    let mut ps = Vec::new();
    for r in 0..n {
        if oof[r].is_finite() {
            ys.push(y[r]);
            ps.push(oof[r]);
        }
    }
    Ok(CvReport {
        fold_r2,
        mean_r2,
        oof_r2: r2(&ys, &ps),
    })
}

/// Convenience baseline used by triage and probes: plain ridge on raw features.
pub fn ridge_baseline_cv(x: &Rows, y: &[f64], k: usize, dataset_key: &str) -> Result<CvReport> {
    let genome = GenomeIr::parse("(pipe X (ridge))")?;
    let _ = RidgeModel::fit(x, y, 0.02); // warm sanity: ensures shapes are valid
    cv_r2(&genome, x, y, k, dataset_key)
}

#[cfg(test)]
#[path = "model/tests.rs"]
mod tests;
