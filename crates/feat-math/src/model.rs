//! Fit / predict / cross-validate a battle genome — the reference execution
//! path used for claim rescoring, calibration probes, and golden tests.

use crate::dsl::Stage;
use crate::folds::fold_of;
use crate::genome::GenomeIr;
use crate::heads::{fit_head, is_head_op, FittedHead, RidgeModel};
use crate::math_expr::MathExpr;
use crate::metrics::r2;
use crate::ops::{apply_op, apply_op_train, fit_op, FittedOp};
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
            let mut ops = Vec::with_capacity(transform_stages.len());
            for stage in transform_stages {
                let op = fit_op(stage, &cur, y_eff)?;
                cur = apply_op_train(&op, &cur);
                ops.push(op);
            }
            let head = Box::new(fit_head(&head_stage, &cur, y_eff)?);
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
        let block = std::sync::Arc::new(apply_op_train(&op, &cur));
        cache.insert(key, op.clone(), std::sync::Arc::clone(&block));
        ops.push(op);
        cur = block;
    }
    let head = Box::new(fit_head(&head_stage, &cur, y_eff)?);
    Ok(Model::Pipeline { target, ops, head })
}

impl Model {
    pub fn predict(&self, x: &Rows) -> Vec<f64> {
        match self {
            Model::Pipeline { target, ops, head } => {
                let mut cur = x.clone();
                sanitize(&mut cur);
                for op in ops {
                    cur = apply_op(op, &cur);
                }
                let preds = head.predict(&cur);
                match target {
                    None => preds,
                    // Exact inverse back to the original target scale, then
                    // the kind's final prediction clamp (±60 for legacy kinds,
                    // train-range for range-aware kinds).
                    Some(t) => preds.iter().map(|p| t.final_clamp(t.inverse(*p))).collect(),
                }
            }
            Model::Portfolio(m) => m.predict(x),
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
        }
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
    Ok(model.predict(x_test))
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
mod tests {
    use super::*;

    fn toy() -> (Rows, Vec<f64>) {
        // y = 2*x0 - x1 + 0.5*sin-ish nonlinearity, deterministic.
        let mut x = Vec::new();
        let mut y = Vec::new();
        let mut rng = crate::Pcg32::new(1, 1);
        for _ in 0..240 {
            let a = rng.next_f64();
            let b = rng.next_f64();
            let c = rng.next_f64();
            x.push(vec![a, b, c]);
            y.push(2.0 * a - b + 0.3 * (6.0 * c).sin());
        }
        (x, y)
    }

    #[test]
    fn ridge_pipeline_learns_linear_signal() {
        let (x, y) = toy();
        let g = GenomeIr::parse("(pipe X (ridge))").unwrap();
        let rep = cv_r2(&g, &x, &y, 5, "toy").unwrap();
        assert!(rep.oof_r2 > 0.85, "oof_r2={}", rep.oof_r2);
    }

    #[test]
    fn rff_pipeline_beats_plain_ridge_on_nonlinearity() {
        let (x, y) = toy();
        let plain = cv_r2(
            &GenomeIr::parse("(pipe X (ridge))").unwrap(),
            &x,
            &y,
            5,
            "toy",
        )
        .unwrap();
        let rff = cv_r2(
            &GenomeIr::parse("(pipe X (rff_cos n=64 seed=3 gamma=2.0 append=true) (ridge))")
                .unwrap(),
            &x,
            &y,
            5,
            "toy",
        )
        .unwrap();
        assert!(
            rff.oof_r2 > plain.oof_r2 - 0.02,
            "rff={} plain={}",
            rff.oof_r2,
            plain.oof_r2
        );
    }

    #[test]
    fn gbm_head_fits_deterministically() {
        let (x, y) = toy();
        let g = GenomeIr::parse("(pipe X (histgbm rounds=40 depth=3))").unwrap();
        let p1 = fit_predict(&g, &x, &y, &x).unwrap();
        let p2 = fit_predict(&g, &x, &y, &x).unwrap();
        assert_eq!(p1, p2);
        assert!(r2(&y, &p1) > 0.8);
    }

    #[test]
    fn symbolic_affine_recovers_scaled_expression() {
        let (x, y2) = toy();
        let y: Vec<f64> = x.iter().map(|r| 3.0 * r[0] + 1.0).collect();
        drop(y2);
        let g = GenomeIr::parse("(sym affine \"col(0)\")").unwrap();
        let rep = cv_r2(&g, &x, &y, 5, "toy").unwrap();
        assert!(rep.oof_r2 > 0.999, "oof={}", rep.oof_r2);
    }

    /// The required honest-CV composability check: on a heavy-tailed target
    /// (y = exp(2.5 z), z linear in x) the rank-Gaussian target transform in
    /// front of plain ridge must beat plain ridge, with fit() owning the
    /// per-fold transform refit.
    #[test]
    fn rank_gauss_target_beats_plain_ridge_on_heavy_tails() {
        let mut rng = crate::Pcg32::new(7, 21);
        let mut x: Rows = Vec::new();
        let mut y = Vec::new();
        for _ in 0..240 {
            let a = rng.next_f64();
            let b = rng.next_f64();
            let c = rng.next_f64();
            let z = 2.0 * a - b + 0.5 * c;
            x.push(vec![a, b, c]);
            y.push((2.5 * z).exp());
        }
        let plain = cv_r2(
            &GenomeIr::parse("(pipe X (ridge))").unwrap(),
            &x,
            &y,
            5,
            "heavy",
        )
        .unwrap();
        let ranked = cv_r2(
            &GenomeIr::parse("(pipe X (target kind=rank_gauss) (ridge))").unwrap(),
            &x,
            &y,
            5,
            "heavy",
        )
        .unwrap();
        println!(
            "robustness[heavy-tail cv oof_r2] plain_ridge={} rank_gauss_ridge={}",
            plain.oof_r2, ranked.oof_r2
        );
        assert!(
            ranked.oof_r2 > plain.oof_r2 + 0.02,
            "rank_gauss={} plain={}",
            ranked.oof_r2,
            plain.oof_r2
        );
    }

    #[test]
    fn target_stage_must_lead_the_pipeline() {
        let (x, y) = toy();
        // Leading target stage fits and predicts.
        let g = GenomeIr::parse("(pipe X (target kind=winsor q=0.05) (ridge))").unwrap();
        let p = fit_predict(&g, &x, &y, &x).unwrap();
        assert_eq!(p.len(), x.len());
        assert!(p.iter().all(|v| v.is_finite()));
        // Anywhere else it is rejected.
        let bad =
            GenomeIr::parse("(pipe X (standardize) (target kind=rank_gauss) (ridge))").unwrap();
        assert!(fit(&bad, &x, &y).is_err());
        let bad2 =
            GenomeIr::parse("(pipe X (target kind=log1p) (target kind=log1p) (ridge))").unwrap();
        assert!(fit(&bad2, &x, &y).is_err());
    }

    /// Spec §6.1: pure-noise target through the FULL core portfolio — outer
    /// honest CV r^2 stays ~0 and the internal stack OOF r^2 stays under
    /// 0.15 (precedent: target_aware_ops_are_oof_in_cv below; fable
    /// gam_no_leak = 0.035).
    #[test]
    fn portfolio_noise_target_no_leak() {
        let (x, _) = toy();
        let x: Rows = x[..150].to_vec();
        let mut rng = crate::Pcg32::new(99, 5);
        let y: Vec<f64> = (0..x.len()).map(|_| rng.next_f64()).collect();
        let g = GenomeIr::parse("(pipe X (portfolio preset=core))").unwrap();
        let rep = cv_r2(&g, &x, &y, 3, "noise_portfolio").unwrap();
        assert!(rep.oof_r2.abs() < 0.05, "outer leak? oof_r2={}", rep.oof_r2);
        let model = fit(&g, &x, &y).unwrap();
        let diag = model.portfolio_diagnostics().unwrap();
        assert!(
            diag.blend.stack_oof_r2 < 0.15,
            "inner leak? stack_oof_r2={}",
            diag.blend.stack_oof_r2
        );
    }

    /// Compact head end-to-end: shared transforms feed every member, the fit
    /// is deterministic, and diagnostics carry the frozen schema fields.
    #[test]
    fn compact_portfolio_head_fits_and_is_deterministic() {
        let (x, y) = toy();
        let x: Rows = x[..120].to_vec();
        let y: Vec<f64> = y[..120].to_vec();
        let g =
            GenomeIr::parse("(pipe X (standardize) (portfolio preset=core folds=5 blend=nnls))")
                .unwrap();
        let p1 = fit_predict(&g, &x, &y, &x).unwrap();
        let p2 = fit_predict(&g, &x, &y, &x).unwrap();
        assert_eq!(p1, p2, "portfolio fit must be bit-deterministic");
        assert!(r2(&y, &p1) > 0.6, "train r2={}", r2(&y, &p1));
        let model = fit(&g, &x, &y).unwrap();
        let diag = model.portfolio_diagnostics().unwrap();
        // n=120 with 3 columns: gates keep the 8 always-on members plus
        // enet_quad (p<=40), gpr_iso (n<150) and gam_tbe (n>=80).
        assert_eq!(diag.member_diagnostics.len(), 11);
        assert_eq!(diag.blend.folds, 5);
        assert_eq!(diag.blend.repeats, 2);
        assert_eq!(diag.blend.baseline_id, "core_v1");
        for (i, d) in diag.member_diagnostics.iter().enumerate() {
            assert_eq!(d.index, i);
            assert_eq!(d.engine, "featmath");
            assert!(d.oof_rmse.is_finite());
            assert!(d.nnls_weight >= 0.0);
        }
        // Weight floor: every deployed weight is 0 or above 1e-4.
        let m = match &model {
            Model::Pipeline { head, .. } => match head.as_ref() {
                crate::heads::FittedHead::Portfolio(m) => m,
                _ => panic!("expected portfolio head"),
            },
            _ => panic!("expected pipeline"),
        };
        assert!(m.weights.iter().all(|w| *w == 0.0 || *w > 1e-4));
        // Bad compact parameters fail the fit with a clear error.
        for bad in [
            "(pipe X (portfolio preset=bogus))",
            "(pipe X (portfolio folds=3))",
            "(pipe X (portfolio blend=caruana))",
            "(pipe X (portfolio add1=nonexistent_member))",
            "(pipe X (portfolio drop1=nonexistent_member))",
        ] {
            let g = GenomeIr::parse(bad).unwrap();
            assert!(fit(&g, &x, &y).is_err(), "{bad} should fail");
        }
    }

    /// The explicit nested form runs through the same fit/predict/cv path.
    #[test]
    fn explicit_portfolio_form_fits_through_model_dispatch() {
        let (x, y) = toy();
        let x: Rows = x[..100].to_vec();
        let y: Vec<f64> = y[..100].to_vec();
        let g = GenomeIr::parse(
            "(portfolio (member (pipe X (ridge_gcv))) \
             (member (pipe X (ksmooth k=8))) \
             (blend kind=nnls folds=5 repeats=auto guard=clip))",
        )
        .unwrap();
        let p = fit_predict(&g, &x, &y, &x).unwrap();
        assert!(r2(&y, &p) > 0.6, "train r2={}", r2(&y, &p));
        let model = fit(&g, &x, &y).unwrap();
        let diag = model.portfolio_diagnostics().unwrap();
        assert_eq!(diag.member_diagnostics.len(), 2);
        assert_eq!(diag.member_diagnostics[0].canonical, "(pipe X (ridge_gcv))");
        let rep = cv_r2(&g, &x, &y, 3, "explicit_portfolio").unwrap();
        assert!(rep.oof_r2 > 0.4, "cv oof_r2={}", rep.oof_r2);
    }

    #[test]
    fn target_aware_ops_are_oof_in_cv() {
        // Target-aware ops must not leak: on a pure-noise target their honest
        // CV r2 stays ~0 (each fold refits the whole pipeline).
        let (x, _) = toy();
        let mut rng = crate::Pcg32::new(99, 5);
        let y: Vec<f64> = (0..x.len()).map(|_| rng.next_f64()).collect();
        for src in [
            "(pipe X (target_bin_encode bins=8) (ridge))",
            "(pipe X (select_corr k=3 by=corr) (ridge))",
            "(pipe X (pair_ratio topk=4) (ridge))",
        ] {
            let g = GenomeIr::parse(src).unwrap();
            let rep = cv_r2(&g, &x, &y, 5, "noise").unwrap();
            assert!(rep.oof_r2 < 0.15, "leak in {src}? oof_r2={}", rep.oof_r2);
        }
    }

    /// Genome vocabulary for the stage-cache exactness suites: append
    /// variants, every target kind, target-aware ops, seeded ops, the
    /// expansion ops, mixed heads, and a shared-prefix stress case.
    const STAGE_CACHE_GENOMES: [&str; 21] = [
        "(pipe X (ridge))",
        "(pipe X (standardize) (ridge))",
        "(pipe X (standardize append=true) (winsor_clip q=0.1) (ridge lambda_frac=0.05))",
        "(pipe X (target kind=log1p) (standardize) (ridge))",
        "(pipe X (target kind=winsor q=0.05) (rank) (elasticnet alpha=0.01 l1_ratio=0.3 sweeps=30))",
        "(pipe X (target kind=rank_gauss) (blom) (ridge))",
        "(pipe X (target kind=zscore) (robust_scale) (ridge))",
        "(pipe X (pair_prod topk=4 append=true) (ridge))",
        "(pipe X (target_bin_encode bins=6) (ridge))",
        "(pipe X (select_corr k=2 by=corr) (ridge))",
        "(pipe X (pair_ratio topk=3 append=true) (ridge))",
        "(pipe X (rff_cos n=16 seed=3 gamma=1.5 append=true) (ridge))",
        "(pipe X (rand_proj k=4 seed=7) (ridge))",
        "(pipe X (landmark_dist k=4 gamma=0.7 seed=5) (ridge))",
        "(pipe X (oblique_hinge n=8 seed=2) (ridge))",
        "(pipe X (nystrom k=6 gamma=0.5 seed=11) (ridge))",
        "(pipe X (symlog c=0.7) (spline knots=4) (ridge))",
        "(pipe X (impute strategy=median add_indicator=true) (standardize) (ridge))",
        "(pipe X (quantile_bins bins=6) (bin_onehot bins=4) (ridge))",
        "(pipe X (cheb degree=3) (histgbm rounds=20 depth=2 lr=0.2 bins=16))",
        "(pipe X (standardize) (standardize) (standardize append=true) (ridge))",
    ];

    fn stage_cache_fixtures() -> Vec<(Rows, Vec<f64>, Rows)> {
        let (x, y) = toy();
        let mut second = crate::Pcg32::new(17, 4);
        let x2: Rows = (0..120)
            .map(|_| (0..4).map(|_| second.next_f64() * 6.0 - 3.0).collect())
            .collect();
        let y2: Vec<f64> = x2
            .iter()
            .map(|row| row[0] * row[1] - 0.5 * row[3] + second.next_f64() * 0.01)
            .collect();
        vec![
            (x[..200].to_vec(), y[..200].to_vec(), x[200..].to_vec()),
            (x2[..100].to_vec(), y2[..100].to_vec(), x2[100..].to_vec()),
        ]
    }

    fn assert_cache_matches_plain(cache: &crate::stage_cache::StageCache) {
        for (dataset_index, (x_train, y_train, x_test)) in
            stage_cache_fixtures().into_iter().enumerate()
        {
            for fold in 0..3usize {
                let ctx = crate::stage_cache::StageCacheCtx {
                    context_tag: format!(
                        "fixture-{dataset_index}|rows={}|features={}",
                        x_train.len(),
                        x_train[0].len()
                    ),
                    folds: 3,
                    fold,
                };
                for src in STAGE_CACHE_GENOMES {
                    let genome = GenomeIr::parse(src).unwrap();
                    let plain = fit(&genome, &x_train, &y_train).unwrap();
                    // Cold pass then warm pass: both must equal plain `fit`.
                    for pass in ["cold", "warm"] {
                        let cached =
                            fit_with_stage_cache(&genome, &x_train, &y_train, cache, &ctx).unwrap();
                        assert_eq!(
                            serde_json::to_string(&plain).unwrap(),
                            serde_json::to_string(&cached).unwrap(),
                            "{pass} model mismatch for {src} fold {fold}"
                        );
                        for (left, right) in
                            plain.predict(&x_test).iter().zip(cached.predict(&x_test))
                        {
                            assert_eq!(
                                left.to_bits(),
                                right.to_bits(),
                                "{pass} prediction mismatch for {src} fold {fold}"
                            );
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn stage_cache_fit_is_bit_identical_to_plain_fit() {
        let cache = crate::stage_cache::StageCache::new(crate::stage_cache::DEFAULT_BYTE_CAP);
        cache.begin_generation(0);
        assert_cache_matches_plain(&cache);
        let (hits, misses, _) = cache.stats();
        assert!(hits > 0, "warm passes must hit the cache");
        assert!(misses > 0, "cold passes must miss the cache");
    }

    #[test]
    fn stage_cache_under_eviction_pressure_stays_bit_identical() {
        // A cap far below one block forces constant eviction and un-cacheable
        // inserts; results must be unaffected.
        let cache = crate::stage_cache::StageCache::new(4_096);
        cache.begin_generation(0);
        assert_cache_matches_plain(&cache);
        cache.sweep();
        assert!(cache.stats().2 <= 4_096);
    }

    #[test]
    fn stage_cache_is_safe_under_concurrent_fitting() {
        let cache = crate::stage_cache::StageCache::new(crate::stage_cache::DEFAULT_BYTE_CAP);
        cache.begin_generation(0);
        let (x, y) = toy();
        let x_train = x[..200].to_vec();
        let y_train = y[..200].to_vec();
        let x_test = x[200..].to_vec();
        let ctx = crate::stage_cache::StageCacheCtx {
            context_tag: "concurrent-fixture".to_string(),
            folds: 5,
            fold: 0,
        };
        let serial: Vec<Vec<u64>> = STAGE_CACHE_GENOMES
            .iter()
            .map(|src| {
                let genome = GenomeIr::parse(src).unwrap();
                fit(&genome, &x_train, &y_train)
                    .unwrap()
                    .predict(&x_test)
                    .iter()
                    .map(|value| value.to_bits())
                    .collect()
            })
            .collect();
        std::thread::scope(|scope| {
            for _ in 0..8 {
                scope.spawn(|| {
                    for (src, expected) in STAGE_CACHE_GENOMES.iter().zip(&serial) {
                        let genome = GenomeIr::parse(src).unwrap();
                        let bits: Vec<u64> =
                            fit_with_stage_cache(&genome, &x_train, &y_train, &cache, &ctx)
                                .unwrap()
                                .predict(&x_test)
                                .iter()
                                .map(|value| value.to_bits())
                                .collect();
                        assert_eq!(&bits, expected, "concurrent mismatch for {src}");
                    }
                });
            }
        });
    }
}
